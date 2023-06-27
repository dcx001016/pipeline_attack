import time
import torch.nn.functional
from communication.comm_utils import *
from modules.dist_gpt_pp_module import *
from data_parallel.dist_dp_utils import get_dp_module
from optimizer.optimizer import get_fp16_optimizer
from compress import get_compressor
import cupy
import wandb

from transformers import get_linear_schedule_with_warmup
from .dist_gpipe_pipeline_async import create_optimizer


class VirtualGPU:
    def __init__(self, args, config, virtual_rank, device,
                 _StageFirst=GPTStageFirst, _StageLast=GPTStageLast,
                 _StageMiddle=GPTStageMiddle):
        if args.fp16:
            self.use_fp16 = True
        else:
            self.use_fp16 = False

        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.pipeline_virtual_gpus = args.pipeline_virtual_gpus
        self.virtual_rank = virtual_rank

        self.device = device

        if virtual_rank == 0:
            self.model = _StageFirst(args, config, device)
        elif virtual_rank == self.pipeline_virtual_gpus - 1:
            self.model = _StageLast(args, config, device)
        else:
            self.model = _StageMiddle(args, config, device)

        

        if self.use_fp16:
            self.model.half()

class BambooVirtualAsync:
    def __init__(self, args, config, device,
                 _StageFirst=GPTStageFirst, _StageLast=GPTStageLast,
                 _StageMiddle=GPTStageMiddle):
        if args.fp16:
            self.use_fp16 = True
            print("=======Gpipe use FP16")
        else:
            self.use_fp16 = False
            print("=======Gpipe use FP32")

        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.global_rank = args.rank
        self.pipeline_group_size = args.pipeline_group_size
        self.pp_rank = get_pipeline_parallel_rank()
        self.pre_node_rank = self.pp_rank - 1
        self.post_node_rank = self.pp_rank + 1 if self.pp_rank != self.pipeline_group_size - 1 else -1
        self.comm = get_pipeline_parallel_comm()
        self.gradient_accumulate_step = args.gradient_accumulate_step
        self.pipeline_virtual_gpus = args.pipeline_virtual_gpus

        assert (args.batch_size % args.micro_batch_size == 0)
        self.micro_batch_num = args.batch_size // args.micro_batch_size
        self.micro_batch_size = args.micro_batch_size
        self.seq_length = args.seq_length
        self.embedding_dim = args.embedding_dim
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_classes = config.num_labels

        self.backward_attack_rate = args.backward_attack_rate

        self.wandb = args.wandb
        self.device = device
        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_send_stream = torch.cuda.Stream(device=device, priority=-1)

        self.forward_recv_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_comp_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        
        self.backward_recv_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_comp_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        
        self._compute_micro_batch_size()

        if self.pp_rank == 0:
            self.input_micro_batches = None
        else:
            self.input_micro_batches = [
                torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                            requires_grad=True, device=self.device, dtype=self.dtype
                           ) for _ in range(self.micro_batch_num)
            ]

        if self.pp_rank == self.pipeline_group_size - 1:
            self.output_micro_batches_grad = None
        else:
            self.output_micro_batches_grad = [
                torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                            requires_grad=False, device=self.device, dtype=self.dtype
                           ) for _ in range(self.micro_batch_num)
            ]

        self.virtual_gpus = [VirtualGPU(args, config, i, device, _StageFirst, _StageLast, _StageMiddle) for i in range(args.virtual_gpus * self.pp_rank, args.virtual_gpus * (self.pp_rank + 1))]

        self.forward_compressor = get_compressor(
            compress_method=args.forward_compress_method, 
            bits=args.forward_bits, 
            bits_act=args.forward_bits_act,
            scale_method=args.forward_scale_method, 
            scale_dims=args.forward_scale_dims,
            max_cache_size=args.max_activation_cache_size,
        )
        self.forward_compressor.build_buffer(
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_length,
            embedding_dim=args.embedding_dim,
            device=device, dtype=self.dtype,
        )
        
        self.backward_compressor = get_compressor(
            compress_method=args.backward_compress_method, 
            bits=args.backward_bits, 
            scale_method=args.backward_scale_method, 
            scale_dims=args.backward_scale_dims,
        )
        self.backward_compressor.build_buffer(
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_length,
            embedding_dim=args.embedding_dim,
            device=device, dtype=self.dtype,
        )

        if self.use_fp16:
            tmp_optimizers = [create_optimizer(self.virtual_gpus[i].model, learning_rate=args.lr, optim=args.optimizer) for i in range(args.virtual_gpus)]
            self.optimizers = [get_fp16_optimizer(args, tmp_optimizers[i], device) for i in range(args.virtual_gpus)]
            self.schedulers = [get_linear_schedule_with_warmup(
                tmp_optimizers[i], args.warmup_steps, args.total_steps, ) for i in range(args.virtual_gpus)]
        else:
            self.optimizers = [create_optimizer(self.virtual_gpus[i].model, learning_rate=args.lr, optim=args.optimizer) for i in range(args.virtual_gpus)]
            self.schedulers = [get_linear_schedule_with_warmup(
                self.optimizers[i], args.warmup_steps, args.total_steps, ) for i in range(args.virtual_gpus)]

        self.global_step = 0

    def _compute_micro_batch_size(self):
        micro_batch_float_num = self.micro_batch_size * self.seq_length * self.embedding_dim
        if self.use_fp16:
            print("=======Current micro-batch send/recv size: {} MB (fp16)"
                  .format(micro_batch_float_num * 2 // 1024 // 1024))
        else:
            print("=======Current micro-batch send/recv size: {} MB (fp32)"
                  .format(micro_batch_float_num*4//1024//1024))
        print("=======Number of micro-batches: {}.".format(self.micro_batch_num))

    def zero_input_grad(self):
        if self.input_micro_batches:
            for input_micro_batch in self.input_micro_batches:
                if input_micro_batch.grad is not None:
                    input_micro_batch.grad.zero_()

    def virtual_forward(self, aux_input_data, index, input_ids_micro_batch=None):
        for i in range(len(self.virtual_gpus)):
            if i == 0:
                if self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1:
                    tmp_output = self.virtual_gpus[i].model(
                        self.input_micro_batches[index], 
                        **{k: v[index] for k, v in aux_input_data.items()}
                    )
                else:
                    tmp_output = self.virtual_gpus[i].model(
                        self.input_micro_batches[index], input_ids=input_ids_micro_batch, 
                        **{k: v[index] for k, v in aux_input_data.items()}
                    )
            else:
                if self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1:
                    tmp_output = self.virtual_gpus[i].model(
                        tmp_output,
                        **{k: v[index] for k, v in aux_input_data.items()}
                    )
                else:
                    tmp_output = self.virtual_gpus[i].model(
                        tmp_output, input_ids=input_ids_micro_batch, 
                        **{k: v[index] for k, v in aux_input_data.items()}
                    )
            if self.virtual_gpus[i].model.training and self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1:
                tmp_output.register_hook(attack_backward_hook(self.backward_attack_rate))
            # if self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1 and (self.virtual_gpus[i].virtual_rank + 1) % len(self.virtual_gpus) == 0:
            #     tmp_output.register_hook(print_tensor_gradient)

        return tmp_output
    
    def virtual_backward():
        pass


    def forward_stage(self, input_data=None, aux_input_data=None):
        if aux_input_data is not None:
            for k in aux_input_data:
                aux_input_data[k] = torch.chunk(aux_input_data[k], self.micro_batch_num, dim=0)
        else:
            aux_input_data = {}

        if self.pp_rank == 0:
            assert(input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        if self.pp_rank == self.pipeline_group_size - 1:
            if input_data is not None:
                input_ids_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
            else:
                input_ids_micro_batches = [None]*self.micro_batch_num
        output_micro_batches = []

        for i in range(self.micro_batch_num):
            if self.pp_rank == 0:
                with torch.cuda.stream(self.torch_comp_stream):
                    current_micro_output = self.virtual_forward(aux_input_data, i)
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.forward_compressor.compress_send(
                        current_micro_output.data, i_micro_batch=i,
                        comm=self.comm, dst=self.post_node_rank, stream=cupy_send_stream
                    )
            elif self.pp_rank == self.pipeline_group_size - 1:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.forward_compressor.recv_decompress(
                        i, comm=self.comm, src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.input_micro_batches[i].data.copy_(_data)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    current_micro_output = self.virtual_forward(aux_input_data, i, input_ids_micro_batches[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.forward_compressor.recv_decompress(
                        i, comm=self.comm, src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.input_micro_batches[i].data.copy_(_data)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    current_micro_output = self.virtual_forward(aux_input_data, i)
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.forward_compressor.compress_send(
                        current_micro_output.data, i_micro_batch=i,
                        comm=self.comm, dst=self.post_node_rank, stream=cupy_send_stream
                    )
            output_micro_batches.append(current_micro_output)

        return output_micro_batches
    
    def backward_stage(self, cached_output_micro_batches: List[torch.Tensor], target=None,
                       loss_func=torch.nn.functional.cross_entropy):
        if self.pp_rank == self.pipeline_group_size - 1:
            assert(target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert(target is None)

        if self.pp_rank == self.pipeline_group_size - 1:
            tr_loss = []

        for i in range(self.micro_batch_num):
            if self.pp_rank == self.pipeline_group_size - 1:
                with torch.cuda.stream(self.torch_comp_stream):
                    loss = loss_func(input=cached_output_micro_batches[i], target=target_as_micro_batches[i])
                    loss.backward()
                    tr_loss.append(loss.item())
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.backward_compressor.compress_send(
                        self.input_micro_batches[i].grad, i_micro_batch=i,
                        comm=self.comm, dst=self.pre_node_rank, stream=cupy_send_stream
                    )
            elif self.pp_rank == 0:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.backward_compressor.recv_decompress(
                            i, comm=self.comm, src=self.post_node_rank, stream=cupy_recv_stream)
                    self.output_micro_batches_grad[i].copy_(_data)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    cached_output_micro_batches[i].backward(gradient=self.output_micro_batches_grad[i])
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.backward_compressor.recv_decompress(
                            i, comm=self.comm, src=self.post_node_rank, stream=cupy_recv_stream)
                    self.output_micro_batches_grad[i].copy_(_data)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    cached_output_micro_batches[i].backward(gradient=self.output_micro_batches_grad[i])
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.backward_compressor.compress_send(
                        self.input_micro_batches[i].grad, i_micro_batch=i,
                        comm=self.comm, dst=self.pre_node_rank, stream=cupy_send_stream
                    )

        if self.pp_rank == self.pipeline_group_size - 1:
            print({
                'loss': sum(tr_loss)/len(tr_loss),
                'lr': self.schedulers[0].get_last_lr()[0],
                'step': self.global_step,
            })

            if self.wandb:
                wandb.log({
                    'loss': sum(tr_loss)/len(tr_loss),
                    'lr': self.schedulers[0].get_last_lr()[0],
                })

        return self.input_micro_batches[0].grad
    
    def optimizer_step(self):
        for i in range(len(self.virtual_gpus)):
            torch.nn.utils.clip_grad_norm_(self.virtual_gpus[i].model.parameters(), 1.0)
        with torch.cuda.stream(self.torch_comp_stream):
            for i in range(len(self.optimizers)):
                self.optimizers[i].step()
                self.schedulers[i].step()

    def sgd_iter(self, input_=None, target=None, sample_ids=None, 
                 aux_input_data=None, loss_func=torch.nn.functional.cross_entropy):
        self.comm.barrier()
        start_time = time.time()
        self.zero_input_grad()
        for i in range(len(self.optimizers)):
            self.optimizers[i].zero_grad(set_to_none=False)

        for step in range(self.gradient_accumulate_step):
            outputs = self.forward_stage(input_, aux_input_data=aux_input_data)
            forward_time = time.time()
            if step == 0:
                forward_slot = forward_time-start_time
            else:
                forward_slot = forward_time-backward_time
            print("Rank {} node forward pass {}/{} takes {:3.2f}s"
                  .format(self.global_rank, step, self.gradient_accumulate_step, forward_slot))
            
            self.comm.barrier()

            grad = self.backward_stage(outputs, target, loss_func=loss_func)
            backward_time = time.time()
            print("Rank {} node backward pass {}/{} takes {:3.2f}s"
                  .format(self.global_rank, step, self.gradient_accumulate_step, backward_time-forward_time))
            
        optimizer_time = time.time()
        self.optimizer_step()
        torch.cuda.synchronize()
        self.comm.barrier()
        end_time = time.time()
        print("Rank {} node optimizer step takes {:3.2f}s".format(self.global_rank, end_time - optimizer_time))
        iter_time = end_time - start_time
        print("Rank {} node whole iteration takes {:3.2f}s".format(self.global_rank, iter_time))
        print("-------------------------------------------")
        self.global_step += 1
        return iter_time, outputs[0], grad
    
    def change_mode(self, mode="train"):
        if mode == "train":
            for i in range(len(self.virtual_gpus)):
                self.virtual_gpus[i].model.train()
        else:
            for i in range(len(self.virtual_gpus)):
                self.virtual_gpus[i].model.eval()

    def infer_stage(self, input_data=None, aux_input_data=None, 
                    labels=None, pred_func=None):
        if aux_input_data is not None:
            for k in aux_input_data:
                aux_input_data[k] = torch.chunk(aux_input_data[k], self.micro_batch_num, dim=0)
        else:
            aux_input_data = {}

        if self.pp_rank == 0:
            assert(input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        if self.pp_rank == self.pipeline_group_size - 1:
            if input_data is not None:
                input_ids_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
            else:
                input_ids_micro_batches = [None]*self.micro_batch_num
            if labels is not None:
                labels = torch.chunk(labels, self.micro_batch_num, dim=0)
            else:
                labels = [None]*self.micro_batch_num

        output_micro_batches = []

        for i in range(self.micro_batch_num):
            if self.pp_rank == 0:
                with torch.cuda.stream(self.torch_comp_stream):
                    current_micro_output = self.virtual_forward(aux_input_data, i)
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
            elif self.pp_rank == self.pipeline_group_size - 1:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    current_micro_output = self.virtual_forward(aux_input_data, i)
                    current_micro_output = pred_func(current_micro_output, labels[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    current_micro_output = self.virtual_forward(aux_input_data, i)
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
                    
            output_micro_batches.append(current_micro_output)
            
        return output_micro_batches

    def infer_iter(self, input_=None, target=None, sample_ids=None, 
                   metrics=None, aux_input_data=None, pred_func=lambda x, y: x.argmax(-1)):
        self.comm.barrier()
        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = self.infer_stage(input_, 
                                       aux_input_data=aux_input_data,
                                       labels=target, pred_func=pred_func)
            if metrics is not None:
                outputs = torch.cat(outputs, 0)
                for metric in metrics:
                    metric.add_batch(predictions=outputs, references=target)
        torch.cuda.synchronize()
        self.comm.barrier()
        


