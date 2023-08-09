import time
import torch.nn.functional
from communication.comm_utils import *
from modules.dist_gpt_pp_module import *
from data_parallel.dist_dp_utils import get_dp_module
from optimizer.optimizer import get_fp16_optimizer
from compress import get_compressor
from utils.common_utils import calculate_metrics
import cupy
import copy
import wandb

from transformers import get_linear_schedule_with_warmup
from .dist_gpipe_pipeline_async import create_optimizer

class CenterServer:
    def __init__(self, args):
        self.micro_batch_num = args.batch_size // args.micro_batch_size
        self.all_output_micro_batches = {i: [None] * self.micro_batch_num for i in range(args.pipeline_virtual_gpus)}
    
    def set_data(self, virtual_stage, index, _data):
        self.all_output_micro_batches[virtual_stage][index] = _data

    def get_data(self, virtual_stage, index):
        return self.all_output_micro_batches[virtual_stage][index]
        

class VirtualGPU:
    def __init__(self, args, config, virtual_rank, device, micro_batch_num,
                 _StageFirst=GPTStageFirst, _StageLast=GPTStageLast,
                 _StageMiddle=GPTStageMiddle):
        if args.fp16:
            self.use_fp16 = True
        else:
            self.use_fp16 = False

        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.pipeline_virtual_gpus = args.pipeline_virtual_gpus
        self.virtual_rank = virtual_rank
        self.micro_batch_num = micro_batch_num
        self.redundant_virtual_rank = virtual_rank - 1 if virtual_rank else args.pipeline_virtual_gpus - 1

        self.device = device

        if virtual_rank == 0:
            self.model = _StageFirst(args, config, device)
        elif virtual_rank == self.pipeline_virtual_gpus - 1:
            self.model = _StageLast(args, config, device)
        else:
            self.model = _StageMiddle(args, config, device)

        if virtual_rank == 0:
            self.input_micro_batches = None
        else:
            self.input_micro_batches = [
                torch.zeros((args.micro_batch_size, args.seq_length, args.embedding_dim),
                            requires_grad=True, device=self.device, dtype=self.dtype
                           ) for _ in range(micro_batch_num)
            ]

        if virtual_rank == self.pipeline_virtual_gpus - 1:
            self.output_micro_batches_grad = None
        else:
            self.output_micro_batches_grad = [
                torch.zeros((args.micro_batch_size, args.seq_length, args.embedding_dim),
                            requires_grad=False, device=self.device, dtype=self.dtype
                           ) for _ in range(self.micro_batch_num)
            ]

        if self.redundant_virtual_rank != 0 and self.redundant_virtual_rank != self.pipeline_virtual_gpus - 1:
            self.redundant_input_micro_batches = [
                torch.zeros((args.micro_batch_size, args.seq_length, args.embedding_dim),
                            requires_grad=True, device=self.device, dtype=self.dtype
                           ) for _ in range(micro_batch_num)
            ]
        else:
            self.redundant_input_micro_batches = None
        self.output_micro_batches = [None] * micro_batch_num

        if self.redundant_virtual_rank != 0 and self.redundant_virtual_rank != self.pipeline_virtual_gpus - 1:
            self.redundant_model = _StageMiddle(args, config, device)
            self.redundant_cached_output_micro_batches = [None] * micro_batch_num

        if self.use_fp16:
            self.model.half()
            self.redundant_model.half() if self.redundant_virtual_rank != 0 and self.redundant_virtual_rank != self.pipeline_virtual_gpus - 1 else None

    def valid(self, aux_input_data, index):
        if not self.model.training or self.redundant_virtual_rank == self.pipeline_virtual_gpus - 1 or self.redundant_virtual_rank == 0:
            return True
            
        self.forward(aux_input_data, index, None, True)
        
        return torch.equal(self.redundant_cached_output_micro_batches[index], self.input_micro_batches[index])
    
    def attack(self):
        pass
    
    def forward(self, aux_input_data, index, input_ids_micro_batch=None, redundant=False):
        if redundant:
            model = self.redundant_model
            virtual_rank = self.redundant_virtual_rank
            input_ = self.redundant_input_micro_batches[index]
        else:
            input_ = self.input_micro_batches[index]
            model = self.model
            virtual_rank = self.virtual_rank
        if virtual_rank == self.pipeline_virtual_gpus - 1:
            out = model(
                input_, input_ids=input_ids_micro_batch,
                **{k: v[index] for k, v in aux_input_data.items()}
            )
        elif virtual_rank == 0:
            out = model(
                input_,
                **{k: v[index] for k, v in aux_input_data.items()}
            )
        else:
            aux_input_data_clone = copy.deepcopy(aux_input_data)
            if "token_type_ids" in aux_input_data_clone:
                del aux_input_data_clone["token_type_ids"]
            out = model(
                input_,
                **{k: v[index] for k, v in aux_input_data_clone.items()}
            )
        if redundant:
            self.redundant_cached_output_micro_batches[index] = out
        else:
            self.output_micro_batches[index] = out
            
        return out.clone()
    
    def backward(self, index: int, redundant=False):
        if redundant:
            if self.redundant_virtual_rank == self.pipeline_virtual_gpus - 1 or self.redundant_virtual_rank == 0:
                return
            return self.redundant_cached_output_micro_batches[index].backward(gradient=self.input_micro_batches[index].grad)
        return self.output_micro_batches[index].backward(gradient=self.output_micro_batches_grad[index])

class SkipLayerVirtualAsync:
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

        self.epoch_metrics = {}
        self.sample_error_times = []

        self.forward_recv_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_comp_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        
        self.backward_recv_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_comp_ready_events = [torch.cuda.Event(enable_timing=False, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.malicious_stage_recv_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.error_stage_recv_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.attack_recv_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.attack_comp_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
        
        self._compute_micro_batch_size()

        self.virtual_gpus = [VirtualGPU(args, config, i, device, self.micro_batch_num, _StageFirst, _StageLast, _StageMiddle) for i in range(args.virtual_gpus * self.pp_rank, args.virtual_gpus * (self.pp_rank + 1))]
        self.center_server = CenterServer(args)

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
            self.optimizers = {self.virtual_gpus[i].virtual_rank: get_fp16_optimizer(args, tmp_optimizers[i], device) for i in range(args.virtual_gpus)}
            self.schedulers = {self.virtual_gpus[i].virtual_rank: get_linear_schedule_with_warmup(tmp_optimizers[i], args.warmup_steps, args.total_steps, ) for i in range(args.virtual_gpus)}
            if self.pp_rank != 0:
                redundant_tmp_optimizers = [create_optimizer(self.virtual_gpus[i].redundant_model, learning_rate=args.lr, optim=args.optimizer) for i in range(args.virtual_gpus)]
                self.redundant_optimizers = {self.virtual_gpus[i].redundant_virtual_rank: get_fp16_optimizer(args, redundant_tmp_optimizers[i], device) for i in range(args.virtual_gpus)}
                self.redundant_schedulers = {self.virtual_gpus[i].redundant_virtual_rank: get_linear_schedule_with_warmup(redundant_tmp_optimizers[i], args.warmup_steps, args.total_steps, ) for i in range(args.virtual_gpus)}
            else:
                redundant_tmp_optimizers = {self.virtual_gpus[i].redundant_virtual_rank: create_optimizer(self.virtual_gpus[i].redundant_model, learning_rate=args.lr, optim=args.optimizer) for i in range(2, args.virtual_gpus)}
                self.redundant_optimizers = {k: get_fp16_optimizer(args, redundant_tmp_optimizers[k], device) for k in redundant_tmp_optimizers}
                self.redundant_schedulers = {k: get_linear_schedule_with_warmup(redundant_tmp_optimizers[k], args.warmup_steps, args.total_steps, ) for k in self.redundant_optimizers}
        else:
            self.optimizers = {self.virtual_gpus[i].virtual_rank: create_optimizer(self.virtual_gpus[i].model, learning_rate=args.lr, optim=args.optimizer) for i in range(args.virtual_gpus)}
            self.schedulers = {k: get_linear_schedule_with_warmup(self.optimizers[k], args.warmup_steps, args.total_steps, ) for k in self.optimizers}
            if self.pp_rank != 0:
                self.redundant_optimizers = {self.virtual_gpus[i].redundant_virtual_rank: create_optimizer(self.virtual_gpus[i].redundant_model, learning_rate=args.lr, optim=args.optimizer) for i in range(args.virtual_gpus)}
                self.redundant_schedulers = {k: get_linear_schedule_with_warmup(self.redundant_optimizers[k], args.warmup_steps, args.total_steps, ) for k in self.redundant_optimizers}
            else:
                self.redundant_optimizers = {self.virtual_gpus[i].redundant_virtual_rank: create_optimizer(self.virtual_gpus[i].redundant_model, learning_rate=args.lr, optim=args.optimizer) for i in range(2, args.virtual_gpus)}
                self.redundant_schedulers = {k: get_linear_schedule_with_warmup(self.redundant_optimizers[k], args.warmup_steps, args.total_steps, ) for k in self.redundant_optimizers}

        self.global_step = 0
        self.get_redundant_models(args.model_name, args.virtual_num_layers)

    def _compute_micro_batch_size(self):
        micro_batch_float_num = self.micro_batch_size * self.seq_length * self.embedding_dim
        if self.use_fp16:
            print("=======Current micro-batch send/recv size: {} MB (fp16)"
                  .format(micro_batch_float_num * 2 // 1024 // 1024))
        else:
            print("=======Current micro-batch send/recv size: {} MB (fp32)"
                  .format(micro_batch_float_num*4//1024//1024))
        print("=======Number of micro-batches: {}.".format(self.micro_batch_num))

    def get_invalid_rate(self):
        return self.invalid_times / self.total_times if self.total_times else 0

    def zero_input_grad(self):
        for gpu in self.virtual_gpus:
            if gpu.input_micro_batches:
                for input_micro_batch in gpu.input_micro_batches:
                    if input_micro_batch.grad is not None:
                        input_micro_batch.grad.zero_()
            
            if gpu.redundant_input_micro_batches:
                for input_micro_batch in gpu.redundant_input_micro_batches:
                    if input_micro_batch.grad is not None:
                        input_micro_batch.grad.zero_()

    def get_metrics(self):
        return

    def forward_attack(self, input: torch.Tensor):
        if self.virtual_gpus[0].model.training:
            input_perturbation = torch.normal(mean=float(input.mean()), std=float(input.std()), size=tuple(input.shape), device=input.device)
            input.data.add_(input_perturbation)

        return input

    def virtual_forward(self, aux_input_data, index: int, input_ids_micro_batch=None, error_stages=None, last_index=-1, last2index=-1):
        for i in range(len(self.virtual_gpus)):
            if error_stages is not None and self.virtual_gpus[i].virtual_rank in error_stages:
                continue
            if self.virtual_gpus[i].virtual_rank != 0:
                self.virtual_gpus[i].input_micro_batches[index].data.copy_(self.center_server.get_data(last_index, index))
            if self.virtual_gpus[i].virtual_rank != 0 and self.virtual_gpus[i].virtual_rank != 1:
                if error_stages is None:
                    self.virtual_gpus[i].redundant_input_micro_batches[index].data.copy_(self.center_server.get_data(last2index, index))
                    valid = self.virtual_gpus[i].valid(aux_input_data, index)
                    if not valid:
                        return torch.full_like(self.virtual_gpus[i].input_micro_batches[index], self.virtual_gpus[i].virtual_rank, dtype=self.dtype, device=self.device), torch.full_like(self.virtual_gpus[i].input_micro_batches[index], self.virtual_gpus[i].virtual_rank, dtype=self.dtype, device=self.device)
                elif self.virtual_gpus[i].redundant_virtual_rank not in error_stages:
                    self.virtual_gpus[i].redundant_input_micro_batches[index].data.copy_(self.center_server.get_data(last2index, index))
                    self.virtual_gpus[i].forward(aux_input_data, index, None, True)
            tmp_output = self.virtual_gpus[i].forward(aux_input_data, index, input_ids_micro_batch)
            if self.virtual_gpus[i].virtual_rank == self.malicious_stage:
                tmp_output = self.forward_attack(tmp_output)
            self.center_server.set_data(self.virtual_gpus[i].virtual_rank, index, tmp_output)
            if last_index != -1:
                last2index = last_index
            last_index = self.virtual_gpus[i].virtual_rank

        return self.center_server.get_data(last_index, index), self.center_server.get_data(last2index, index) if last2index != -1 else torch.zeros_like(self.center_server.get_data(last_index, index), dtype=self.dtype, device=self.device)
    
    def virtual_backward(self, loss_func, index: int, target_as_micro_batch=None, tr_loss=None, error_stages=None):
        for i in reversed(range(len(self.virtual_gpus))):
            if error_stages is not None and self.virtual_gpus[i].virtual_rank in error_stages:
                continue
            if self.virtual_gpus[i].virtual_rank == self.pipeline_virtual_gpus - 1:
                loss = loss_func(input=self.virtual_gpus[i].output_micro_batches[index], target=target_as_micro_batch)
                loss.backward()
                tr_loss.append(loss.item())
            else:
                self.virtual_gpus[i].backward(index)
            
            if error_stages is None or self.virtual_gpus[i].redundant_virtual_rank not in error_stages:
                self.virtual_gpus[i].backward(index, True)
            last_index = i
            for j in reversed(range(0, i)):
                if error_stages is None or self.virtual_gpus[j].virtual_rank not in error_stages:
                    self.virtual_gpus[j].output_micro_batches_grad[index].data.copy_(self.virtual_gpus[i].input_micro_batches[index].grad)
                    break
        return self.virtual_gpus[last_index].input_micro_batches[index].grad if self.virtual_gpus[last_index].virtual_rank else None

    def forward_stage(self, input_data=None, aux_input_data=None, error_stages=None):
        if aux_input_data is not None:
            for k in aux_input_data:
                aux_input_data[k] = torch.chunk(aux_input_data[k], self.micro_batch_num, dim=0)
        else:
            aux_input_data = {}

        if self.pp_rank == 0:
            assert(input_data is not None)
            self.virtual_gpus[0].input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        if self.pp_rank == self.pipeline_group_size - 1:
            if input_data is not None:
                input_ids_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
            else:
                input_ids_micro_batches = [None]*self.micro_batch_num
        output_micro_batches = []

        last_index = -1
        last2index = -1
        if self.virtual_gpus[0].virtual_rank != 0:
            if error_stages is not None:
                for i in reversed(range(self.virtual_gpus[0].virtual_rank)):
                    if last_index == -1 and i not in error_stages:
                        last_index = i
                    if last2index == -1 and (i - 1) not in error_stages:
                        last2index = i - 1
                    if last_index != -1 and last2index != -1:
                        break
            else:
                last_index = self.virtual_gpus[0].virtual_rank - 1
                last2index = self.virtual_gpus[0].virtual_rank - 2

        for i in range(self.micro_batch_num):
            if self.pp_rank == 0:
                with torch.cuda.stream(self.torch_comp_stream):
                    current_micro_output, last_input = self.virtual_forward(aux_input_data, i, error_stages=error_stages, last_index=last_index, last2index=last2index)
                    if current_micro_output.unique().size(0) == 1:
                        self.error_stage.data = torch.tensor(int(current_micro_output.unique()[0]), dtype=torch.int, device=self.device)
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
                    self.center_server.set_data(last_index, i, _data[0])
                    if last2index != -1:
                        self.center_server.set_data(last2index, i, _data[1])
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    if _data.unique().size(0) != 1:
                        current_micro_output, last_input = self.virtual_forward(aux_input_data, i, input_ids_micro_batches[i], error_stages=error_stages, last_index=last_index, last2index=last2index)
                    else:
                        current_micro_output, last_input = _data[0].clone(), _data[1].clone()
                    if current_micro_output.unique().size(0) == 1:
                        self.error_stage.data = torch.tensor(int(current_micro_output.unique()[0]), dtype=torch.int, device=self.device)
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.forward_compressor.recv_decompress(
                        i, comm=self.comm, src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.center_server.set_data(last_index, i, _data[0])
                    if last2index != -1:
                        self.center_server.set_data(last2index, i, _data[1])
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    if _data.unique().size(0) != 1:
                        current_micro_output, last_input = self.virtual_forward(aux_input_data, i, error_stages=error_stages, last_index=last_index, last2index=last2index)
                    else:
                        current_micro_output, last_input = _data[0].clone(), _data[1].clone()
                    if current_micro_output.unique().size(0) == 1:
                        self.error_stage.data = torch.tensor(int(current_micro_output.unique()[0]), dtype=torch.int, device=self.device)
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
    
    def backward_stage(self, target=None,
                       loss_func=torch.nn.functional.cross_entropy, error_stages=None):
        if self.pp_rank == self.pipeline_group_size - 1:
            assert(target is not None)
            target_as_micro_batches = torch.chunk(target, self.micro_batch_num, dim=0)
        else:
            assert(target is None)

        if self.pp_rank == self.pipeline_group_size - 1:
            tr_loss = []
        else:
            if error_stages is None:
                last_index = len(self.virtual_gpus) - 1
            else:
                last_index = -1
                for i in reversed(range(len(self.virtual_gpus))):
                    if self.virtual_gpus[i].virtual_rank not in error_stages:
                        last_index = i
                        break

        for i in range(self.micro_batch_num):
            if self.pp_rank == self.pipeline_group_size - 1:
                with torch.cuda.stream(self.torch_comp_stream):
                    grad = self.virtual_backward(loss_func, i, target_as_micro_batches[i], tr_loss, error_stages)
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.backward_compressor.compress_send(
                        grad, i_micro_batch=i,
                        comm=self.comm, dst=self.pre_node_rank, stream=cupy_send_stream
                    )
            elif self.pp_rank == 0:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.backward_compressor.recv_decompress(
                            i, comm=self.comm, src=self.post_node_rank, stream=cupy_recv_stream)
                    self.virtual_gpus[last_index].output_micro_batches_grad[i].copy_(_data)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    grad = self.virtual_backward(loss_func, i, error_stages=error_stages)
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.backward_compressor.recv_decompress(
                            i, comm=self.comm, src=self.post_node_rank, stream=cupy_recv_stream)
                    if last_index != -1:
                        self.virtual_gpus[last_index].output_micro_batches_grad[i].copy_(_data)
                    self.torch_recv_stream.record_event(self.backward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.backward_recv_ready_events[i])
                    if last_index != -1:
                        grad = self.virtual_backward(loss_func, i, error_stages=error_stages)
                    else:
                        grad = _data
                    self.torch_comp_stream.record_event(self.backward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.backward_comp_ready_events[i])
                    self.backward_compressor.compress_send(
                        grad, i_micro_batch=i,
                        comm=self.comm, dst=self.pre_node_rank, stream=cupy_send_stream
                    )

        if self.pp_rank == self.pipeline_group_size - 1:
            print({
                'loss': sum(tr_loss)/len(tr_loss) if len(tr_loss) else 0,
                'lr': self.schedulers[self.virtual_gpus[0].virtual_rank].get_last_lr()[0],
                'step': self.global_step,
            })

            if self.wandb:
                wandb.log({
                    'loss': sum(tr_loss)/len(tr_loss) if len(tr_loss) else 0,
                    'lr': self.schedulers[self.virtual_gpus[0].virtual_rank].get_last_lr()[0],
                })

        return grad
    
    def get_malicious_stage(self):
        if self.pp_rank == 0:
            with torch.cuda.stream(self.torch_send_stream):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                if self.pipeline_virtual_gpus > 2 and random.random() < self.forward_attack_rate:
                    self.malicious_stage.data = torch.tensor(random.randint(1, self.pipeline_virtual_gpus - 2), dtype=torch.int, device=self.device)
                self.comm.send(self.malicious_stage, self.post_node_rank, cupy_send_stream)
        elif self.pp_rank == self.pipeline_group_size - 1:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.comm.recv(self.malicious_stage, self.pre_node_rank, cupy_recv_stream)
        else:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.comm.recv(self.malicious_stage, self.pre_node_rank, cupy_recv_stream)
                self.torch_recv_stream.record_event(self.malicious_stage_recv_ready_event)
            with torch.cuda.stream(self.torch_send_stream):
                self.torch_send_stream.wait_event(self.malicious_stage_recv_ready_event)
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.comm.send(self.malicious_stage, self.post_node_rank, cupy_send_stream)

    
    def get_error_stage(self):
        if self.pp_rank == self.pipeline_group_size - 1:
            with torch.cuda.stream(self.torch_send_stream):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.comm.send(self.error_stage, self.pre_node_rank, cupy_send_stream)
        elif self.pp_rank == 0:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.comm.recv(self.error_stage, self.post_node_rank, cupy_recv_stream)
        else:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.comm.recv(self.error_stage, self.post_node_rank, cupy_recv_stream)
                self.torch_recv_stream.record_event(self.error_stage_recv_ready_event)
            with torch.cuda.stream(self.torch_send_stream):
                self.torch_send_stream.wait_event(self.error_stage_recv_ready_event)
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.comm.send(self.error_stage, self.pre_node_rank, cupy_send_stream)

    def get_redundant_models(self, model_name, virtual_num_layers):
        for gpu in self.virtual_gpus:
            if hasattr(gpu, "redundant_model"):
                _i = gpu.redundant_virtual_rank * virtual_num_layers
                for i in range(len(gpu.redundant_model.model)):
                    gpu.redundant_model.model[i].load_state_dict(
                        torch.load(f'{model_name}/pytorch_{i + _i}.pt')
                    )

    def optimizer_step(self, frozen_stages=[]):
        for i in range(len(self.virtual_gpus)):
            torch.nn.utils.clip_grad_norm_(self.virtual_gpus[i].model.parameters(), 1.0)
            if hasattr(self.virtual_gpus[i], "redundant_model"):
                torch.nn.utils.clip_grad_norm_(self.virtual_gpus[i].redundant_model.parameters(), 1.0)

        with torch.cuda.stream(self.torch_comp_stream):
            for k in self.optimizers:
                if k not in frozen_stages:
                    self.optimizers[k].step()
                    self.schedulers[k].step()
            for k in self.redundant_optimizers:
                if k not in frozen_stages:
                    self.redundant_optimizers[k].step()
                    self.redundant_schedulers[k].step()

    def sgd_iter(self, input_=None, target=None, sample_ids=None, 
                 aux_input_data=None, loss_func=torch.nn.functional.cross_entropy,
                 iter=0):
        self.comm.barrier()
        start_time = time.time()
        self.zero_input_grad()
        for i in self.optimizers:
            self.optimizers[i].zero_grad(set_to_none=False)
        for i in self.redundant_optimizers:
            self.redundant_optimizers[i].zero_grad(set_to_none=False)

        for step in range(self.gradient_accumulate_step):
            self.malicious_stage = torch.tensor(-1, dtype=torch.int, device=self.device)
            self.error_stage = torch.tensor(-1, dtype=torch.int, device=self.device)
            self.get_malicious_stage()
            torch.cuda.synchronize()
            self.comm.barrier()
            outputs = self.forward_stage(input_, aux_input_data=aux_input_data)
            forward_time = time.time()
            if step == 0:
                forward_slot = forward_time-start_time
            else:
                forward_slot = forward_time-backward_time
            # print("Rank {} node forward pass {}/{} takes {:3.2f}s"
                #   .format(self.global_rank, step, self.gradient_accumulate_step, forward_slot))
            
            self.comm.barrier()

            self.get_error_stage()
            torch.cuda.synchronize()
            self.comm.barrier()

            if self.error_stage != -1:
                self.invalid_times += 1
                error_stages = [stage_index for stage_index in range(max(1, int(self.error_stage) - 1), min(self.pipeline_virtual_gpus - 2, int(self.error_stage) + 1))]
                frozen_stages = [stage_index for stage_index in range(max(1, int(self.error_stage) - 2), min(self.pipeline_virtual_gpus - 2, int(self.error_stage) + 1))]
                outputs = self.forward_stage(input_, aux_input_data=aux_input_data, error_stages=error_stages)
            else:
                error_stages = None
                frozen_stages = []

            grad = self.backward_stage(target, loss_func=loss_func, error_stages=error_stages)
            backward_time = time.time()
            # print("Rank {} node backward pass {}/{} takes {:3.2f}s"
            #       .format(self.global_rank, step, self.gradient_accumulate_step, backward_time-forward_time))
            
        optimizer_time = time.time()
        self.optimizer_step(frozen_stages)
        # print("Rank {} node optimizer_step".format(self.global_rank))
        torch.cuda.synchronize()
        # print("Rank {} node synchronize".format(self.global_rank))
        self.comm.barrier()
        end_time = time.time()
        # print("Rank {} node optimizer step takes {:3.2f}s".format(self.global_rank, end_time - optimizer_time))
        iter_time = end_time - start_time
        print("Rank {} node whole iteration takes {:3.2f}s".format(self.global_rank, iter_time))
        print("-------------------------------------------")
        self.global_step += 1
        self.total_times += 1
        return iter_time, outputs[0], grad
    
    def change_mode(self, mode="train"):
        self.mode = mode
        if mode == "train":
            for i in range(len(self.virtual_gpus)):
                self.virtual_gpus[i].model.train()
                if hasattr(self.virtual_gpus[i], "redundant_model"):
                    self.virtual_gpus[i].redundant_model.train()
        else:
            for i in range(len(self.virtual_gpus)):
                self.virtual_gpus[i].model.eval()
                if hasattr(self.virtual_gpus[i], "redundant_model"):
                    self.virtual_gpus[i].redundant_model.eval()

    def infer_stage(self, input_data=None, aux_input_data=None, 
                    labels=None, pred_func=None):
        if aux_input_data is not None:
            for k in aux_input_data:
                aux_input_data[k] = torch.chunk(aux_input_data[k], self.micro_batch_num, dim=0)
        else:
            aux_input_data = {}

        if self.pp_rank == 0:
            assert(input_data is not None)
            self.virtual_gpus[0].input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
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
        last_index = self.virtual_gpus[0].virtual_rank - 1
        last2index = self.virtual_gpus[0].virtual_rank - 2

        for i in range(self.micro_batch_num):
            if self.pp_rank == 0:
                with torch.cuda.stream(self.torch_comp_stream):
                    current_micro_output, _ = self.virtual_forward(aux_input_data, i, last_index=last_index, last2index=last2index)
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
            elif self.pp_rank == self.pipeline_group_size - 1:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.comm.recv(self.center_server.all_output_micro_batches[self.virtual_gpus[0].virtual_rank-1][i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    current_micro_output, _ = self.virtual_forward(aux_input_data, i, input_ids_micro_batches[i], last_index=last_index, last2index=last2index)
                    current_micro_output = pred_func(current_micro_output, labels[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.comm.recv(self.center_server.all_output_micro_batches[self.virtual_gpus[0].virtual_rank-1][i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    current_micro_output, _ = self.virtual_forward(aux_input_data, i, last_index=last_index, last2index=last2index)
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
        


