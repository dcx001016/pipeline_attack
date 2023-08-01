import time
import torch.nn.functional
from communication.comm_utils import *
from modules.dist_gpt_pp_module import *
from optimizer.optimizer import get_fp16_optimizer
from compress import get_compressor
from utils.common_utils import calculate_metrics
import cupy
import copy
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
        self.history_length = args.history_length
        self.top_n = args.top_n
        self.history = []
        self.distance_mode = args.distance

        self.device = device

        if virtual_rank == 0:
            self.model = _StageFirst(args, config, device)
        elif virtual_rank == self.pipeline_virtual_gpus - 1:
            self.model = _StageLast(args, config, device)
        else:
            self.model = _StageMiddle(args, config, device)

        if self.use_fp16:
            self.model.half()

    def forward(self, input, aux_input_data, index, input_ids_micro_batch=None):
        if self.virtual_rank == self.pipeline_virtual_gpus - 1:
            out = self.model(
                input, input_ids=input_ids_micro_batch,
                **{k: v[index] for k, v in aux_input_data.items()}
            )
        elif self.virtual_rank == 0:
            out = self.model(
                input,
                **{k: v[index] for k, v in aux_input_data.items()}
            )
        else:
            aux_input_data_clone = copy.deepcopy(aux_input_data)
            if "token_type_ids" in aux_input_data_clone:
                del aux_input_data_clone["token_type_ids"]
            out = self.model(
                input,
                **{k: v[index] for k, v in aux_input_data_clone.items()}
            )
            
        return out

    def valid(self, _in, _out):
        if not self.model.training:
            return True
        # return True
        _in = _in.float()
        if len(self.history) < self.history_length:
            self.history.append([_in, _out])
            return True
        
        if self.distance_mode == "l2":
            in_distances = [torch.dist(history[0], _in) for history in self.history]
        elif self.distance_mode == "cos":
            in_distances = [1 - torch.nn.functional.cosine_similarity(history[0].view(1, -1), _in.view(1, -1)) for history in self.history]
        else:
            print("Not recognize this distance.")
            assert False

        in_closest_index = min(range(len(in_distances)), key=lambda i: in_distances[i])

        if self.distance_mode == "l2":
            out_distances = [torch.dist(history[1], _out) for history in self.history]
        elif self.distance_mode == "cos":
            out_distances = [1 - torch.nn.functional.cosine_similarity(history[1].view(1, -1), _out.view(1, -1)) for history in self.history]
        else:
            print("Not recognize this distance.")
            assert False

        out_closest_indexes = sorted(range(len(out_distances)), key=lambda i: out_distances[i])[:self.top_n]
        
        if in_closest_index in out_closest_indexes:
            del self.history[0]
            self.history.append([_in, _out])
            return True
        
        del self.history[in_closest_index]
        self.history.append([_in, _out])
        return False
        

class VirtualKeyValueAsync:
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

        self.forward_attack_rate = args.forward_attack_rate
        self.backward_attack_rate = args.backward_attack_rate
        self.invalid_times = 0
        self.total_times = 0

        self.wandb = args.wandb
        self.device = device
        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_send_stream = torch.cuda.Stream(device=device, priority=-1)

        self.local_attack = []
        self.local_invalid = []
        self.global_invalid = []
        self.global_attack = []
        self.epoch_metrics = {}
        self.sample_error_times = []
        self.attack_stage_cache = torch.zeros(self.micro_batch_num, dtype=torch.int, device=self.device)

        self.forward_recv_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_comp_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        
        self.backward_recv_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_comp_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.status_recv_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.attack_recv_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.attack_comp_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
        
        
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
            redundant=2
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

    def get_metrics(self):
        epoch_metrics = calculate_metrics(self.local_attack, self.local_invalid)
        self.epoch_metrics[len(self.epoch_metrics)] = {
            "tp": epoch_metrics[0],
            "fp": epoch_metrics[1],
            "tn": epoch_metrics[2],
            "fn": epoch_metrics[3]
        }
        self.local_attack = []
        self.local_invalid = []

    def forward_attack(self, input: torch.Tensor, last_input: torch.Tensor, index):
        p = random.random()
        if self.virtual_gpus[0].model.training and p < self.forward_attack_rate:
            input_perturbation = torch.normal(mean=float(input.mean()), std=float(input.std()), size=tuple(input.shape), device=input.device)
            input.data.add_(input_perturbation)
            last_input_min = last_input.min()
            last_input_max = last_input.max()
            last_input_perturbation = torch.normal(mean=float(last_input.float().mean()), std=float(last_input.float().std()), size=tuple(last_input.shape), device=last_input.device)
            if last_input.dtype == torch.int64:
                last_input_perturbation = last_input_perturbation.round().to(torch.long)
            last_input.data.add_(last_input_perturbation)
            last_input.data.clamp_(last_input_min, last_input_max)
            self.attack_stage[index] = 1
        return input, last_input

    def virtual_forward(self, aux_input_data, index, input_ids_micro_batch=None, last_input=None):
        for i in range(len(self.virtual_gpus)):
            if i == 0:
                if self.virtual_gpus[i].virtual_rank == 0:
                    last_input = self.input_micro_batches[index].clone()
                    tmp_output = self.virtual_gpus[i].forward(self.input_micro_batches[index], aux_input_data, index, input_ids_micro_batch)
                    tmp_output, last_input = self.forward_attack(tmp_output, last_input, index)
                elif self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1:
                    with torch.no_grad():
                        if last_input is not None and not self.virtual_gpus[i].valid(last_input.cpu(), self.input_micro_batches[index].cpu()):
                            return torch.zeros_like(self.input_micro_batches[index], device=self.device), torch.zeros_like(self.input_micro_batches[index], device=self.device)
                    last_input = self.input_micro_batches[index].clone()
                    tmp_output = self.virtual_gpus[i].forward(self.input_micro_batches[index], aux_input_data, index, input_ids_micro_batch)
                    tmp_output, last_input = self.forward_attack(tmp_output, last_input, index)
                else:
                    with torch.no_grad():
                        if last_input is not None and not self.virtual_gpus[i].valid(last_input.cpu(), self.input_micro_batches[index].cpu()):
                            return torch.zeros_like(self.input_micro_batches[index], device=self.device), torch.zeros_like(self.input_micro_batches[index], device=self.device)
                    last_input = self.input_micro_batches[index].clone()
                    tmp_output = self.virtual_gpus[i].forward(self.input_micro_batches[index], aux_input_data, index, input_ids_micro_batch)
            else:
                if self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1:
                    with torch.no_grad():
                        if not self.virtual_gpus[i].valid(last_input.cpu(), tmp_output.cpu()):
                            return torch.zeros_like(tmp_output, device=self.device), torch.zeros_like(tmp_output, device=self.device)
                    last_input = tmp_output.clone()
                    tmp_output = self.virtual_gpus[i].forward(tmp_output, aux_input_data, index, input_ids_micro_batch)
                    tmp_output, last_input = self.forward_attack(tmp_output, last_input, index)
                else:
                    with torch.no_grad():
                        if not self.virtual_gpus[i].valid(last_input.cpu(), tmp_output.cpu()):
                            return torch.zeros_like(tmp_output, device=self.device), torch.zeros_like(tmp_output, device=self.device)
                    last_input = tmp_output.clone()
                    tmp_output = self.virtual_gpus[i].forward(tmp_output, aux_input_data, index, input_ids_micro_batch)

            # if self.virtual_gpus[i].model.training and self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1:
            #     tmp_output.register_hook(attack_backward_hook(self.backward_attack_rate))
            # if self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1 and (self.virtual_gpus[i].virtual_rank + 1) % len(self.virtual_gpus) == 0:
            #     tmp_output.register_hook(print_tensor_gradient)
        return tmp_output, last_input
    
    def virtual_backward():
        pass

    def get_invalid_rate(self):
        return self.invalid_times / self.total_times

    def forward_stage(self, input_data=None, aux_input_data=None, iter=0):
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
                    current_micro_output, last_input = self.virtual_forward(aux_input_data, i)
                    if torch.all(current_micro_output == 0):
                        self.success_stage[i] = 0
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    send_data = torch.stack((current_micro_output.data, last_input.data), dim=0)
                    self.forward_compressor.compress_send(
                        send_data.data, i_micro_batch=i,
                        comm=self.comm, dst=self.post_node_rank, stream=cupy_send_stream
                    )
            elif self.pp_rank == self.pipeline_group_size - 1:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.forward_compressor.recv_decompress(
                        i, comm=self.comm, src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.input_micro_batches[i].data.copy_(_data[0])
                    last_input = _data[1].clone()
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    if torch.any(_data):
                        current_micro_output, last_input = self.virtual_forward(aux_input_data, i, input_ids_micro_batches[i], last_input)
                    else:
                        current_micro_output, last_input = _data[0].clone(), _data[1].clone()
                    if torch.all(current_micro_output == 0):
                        self.success_stage[i] = 0
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.forward_compressor.recv_decompress(
                        i, comm=self.comm, src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.input_micro_batches[i].data.copy_(_data[0])
                    last_input = _data[1].clone()
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    if torch.any(_data):
                        current_micro_output, last_input = self.virtual_forward(aux_input_data, i, last_input=last_input)
                    else:
                        current_micro_output, last_input = _data[0].clone(), _data[1].clone()
                    if torch.all(current_micro_output == 0):
                        self.success_stage[i] = 0
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    send_data = torch.stack((current_micro_output.data, last_input.data), dim=0)
                    self.forward_compressor.compress_send(
                        send_data.data, i_micro_batch=i,
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
            if self.success_stage[i] == 0:
                continue
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
                'loss': sum(tr_loss)/len(tr_loss) if len(tr_loss) else 0,
                'lr': self.schedulers[0].get_last_lr()[0],
                'step': self.global_step,
            })

            if self.wandb:
                wandb.log({
                    'loss': sum(tr_loss)/len(tr_loss) if len(tr_loss) else 0,
                    'lr': self.schedulers[0].get_last_lr()[0],
                })

        return self.input_micro_batches[0].grad
    
    def get_status(self):
        if self.pp_rank == self.pipeline_group_size - 1:
            with torch.cuda.stream(self.torch_send_stream):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.comm.send(self.success_stage, self.pre_node_rank, cupy_send_stream)
        elif self.pp_rank == 0:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.comm.recv(self.success_stage, self.post_node_rank, cupy_recv_stream)
        else:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.comm.recv(self.success_stage, self.post_node_rank, cupy_recv_stream)
                self.torch_recv_stream.record_event(self.status_recv_ready_event)
            with torch.cuda.stream(self.torch_send_stream):
                self.torch_send_stream.wait_event(self.status_recv_ready_event)
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.comm.send(self.success_stage, self.pre_node_rank, cupy_send_stream)

    def get_attack_status(self):
        if self.pp_rank == 0:
            with torch.cuda.stream(self.torch_send_stream):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.comm.send(self.attack_stage, self.post_node_rank, cupy_send_stream)
        elif self.pp_rank == self.pipeline_group_size - 1:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.comm.recv(self.attack_stage_cache, self.pre_node_rank, cupy_recv_stream)
                self.torch_recv_stream.record_event(self.attack_recv_ready_event)
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.wait_event(self.attack_recv_ready_event)
                for i in range(len(self.attack_stage_cache)):
                    if self.attack_stage_cache[i]:
                        self.attack_stage[i] = self.attack_stage_cache[i]
        else:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.comm.recv(self.attack_stage_cache, self.pre_node_rank, cupy_recv_stream)
                self.torch_recv_stream.record_event(self.attack_recv_ready_event)
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.wait_event(self.attack_recv_ready_event)
                for i in range(len(self.attack_stage_cache)):
                    if self.attack_stage_cache[i]:
                        self.attack_stage[i] = self.attack_stage_cache[i]
                self.torch_comp_stream.record_event(self.attack_comp_ready_event)
            with torch.cuda.stream(self.torch_send_stream):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.torch_send_stream.wait_event(self.attack_comp_ready_event)
                self.comm.send(self.attack_stage, self.post_node_rank, cupy_send_stream)
                

    def optimizer_step(self):
        for i in range(len(self.virtual_gpus)):
            torch.nn.utils.clip_grad_norm_(self.virtual_gpus[i].model.parameters(), 1.0)
        with torch.cuda.stream(self.torch_comp_stream):
            for i in range(len(self.optimizers)):
                self.optimizers[i].step()
                self.schedulers[i].step()

    def sgd_iter(self, input_=None, target=None, sample_ids=None, 
                 aux_input_data=None, loss_func=torch.nn.functional.cross_entropy,
                 iter=0):
        self.comm.barrier()
        start_time = time.time()
        self.zero_input_grad()
        for i in range(len(self.optimizers)):
            self.optimizers[i].zero_grad(set_to_none=False)

        for step in range(self.gradient_accumulate_step):
            self.success_stage = torch.ones(self.micro_batch_num, dtype=torch.int, device=self.device)
            self.attack_stage = torch.zeros(self.micro_batch_num, dtype=torch.int, device=self.device)
            outputs = self.forward_stage(input_, aux_input_data=aux_input_data, iter=iter)
            forward_time = time.time()
            if step == 0:
                forward_slot = forward_time-start_time
            else:
                forward_slot = forward_time-backward_time
            # print("Rank {} node forward pass {}/{} takes {:3.2f}s"
            #       .format(self.global_rank, step, self.gradient_accumulate_step, forward_slot))
            
            self.comm.barrier()

            self.get_status()
            self.get_attack_status()
            torch.cuda.synchronize()
            self.comm.barrier()

            self.global_attack.extend(self.attack_stage.tolist())
            self.global_invalid.extend([1 if x == 0 else 0 for x in self.success_stage])
            self.local_attack.extend(self.attack_stage.tolist())
            self.local_invalid.extend([1 if x == 0 else 0 for x in self.success_stage])
            if len(self.sample_error_times) // self.micro_batch_num == iter:
                self.sample_error_times.extend([0 if self.attack_stage[i] != self.success_stage[i] else 1 for i in range(self.micro_batch_num)])
            else:
                for i in range(self.micro_batch_num):
                    if self.attack_stage[i] == self.success_stage[i]:
                        self.sample_error_times[self.micro_batch_num * iter + i] += 1


            grad = self.backward_stage(outputs, target, loss_func=loss_func)
            backward_time = time.time()
            # print("Rank {} node backward pass {}/{} takes {:3.2f}s"
            #       .format(self.global_rank, step, self.gradient_accumulate_step, backward_time-forward_time))
            
        optimizer_time = time.time()
        self.optimizer_step()
        torch.cuda.synchronize()
        # print("Rank {} node optimizer step".format(self.global_rank))
        self.comm.barrier()
        end_time = time.time()
        # print("Rank {} node optimizer step takes {:3.2f}s".format(self.global_rank, end_time - optimizer_time))
        iter_time = end_time - start_time
        print("Rank {} node whole iteration takes {:3.2f}s".format(self.global_rank, iter_time))
        print("-------------------------------------------")
        self.global_step += 1
        self.total_times += len(self.success_stage)
        self.invalid_times += torch.sum(self.success_stage == 0).item()
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
                    current_micro_output, _ = self.virtual_forward(aux_input_data, i)
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
                    current_micro_output, _ = self.virtual_forward(aux_input_data, i)
                    current_micro_output = pred_func(current_micro_output, labels[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    current_micro_output, _ = self.virtual_forward(aux_input_data, i)
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
        


