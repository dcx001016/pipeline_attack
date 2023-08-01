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

        self.input_micro_batches = [None] * micro_batch_num
        self.tmp_out = [None] * micro_batch_num

        if self.redundant_virtual_rank == 0:
            self.redundant_model = _StageFirst(args, config, device)
            self.redundant_cached_output_micro_batches = [None] * micro_batch_num
        elif self.redundant_virtual_rank != self.pipeline_virtual_gpus - 1:
            self.redundant_model = _StageMiddle(args, config, device)
            self.redundant_cached_output_micro_batches = [None] * micro_batch_num

        if self.use_fp16:
            self.model.half()
            self.redundant_model.half() if self.redundant_virtual_rank != self.pipeline_virtual_gpus - 1 else None

    def valid(self, last_input, last2output, input, aux_input_data, index):
        if not self.model.training or self.redundant_virtual_rank == self.pipeline_virtual_gpus - 1:
            return True, torch.zeros_like(input)
        
        if not torch.equal(last2output, last_input):
            return False, torch.zeros_like(input)
            
        self.redundant_cached_output_micro_batches[index] = self.forward(last_input, aux_input_data, index, None, True)
        
        if torch.equal(self.redundant_cached_output_micro_batches[index], input):
            return True, self.redundant_cached_output_micro_batches[index]
        return False, torch.zeros_like(self.redundant_cached_output_micro_batches[index])
    
    def attack(self):
        pass
    
    def forward(self, input, aux_input_data, index, input_ids_micro_batch=None, redundant=False):
        if redundant:
            model = self.redundant_model
            virtual_rank = self.redundant_virtual_rank
        else:
            self.input_micro_batches[index] = input
            model = self.model
            virtual_rank = self.virtual_rank
        if virtual_rank == self.pipeline_virtual_gpus - 1:
            out = model(
                input, input_ids=input_ids_micro_batch,
                **{k: v[index] for k, v in aux_input_data.items()}
            )
        elif virtual_rank == 0:
            out = model(
                input,
                **{k: v[index] for k, v in aux_input_data.items()}
            )
        else:
            aux_input_data_clone = copy.deepcopy(aux_input_data)
            if "token_type_ids" in aux_input_data_clone:
                del aux_input_data_clone["token_type_ids"]
            out = model(
                input,
                **{k: v[index] for k, v in aux_input_data_clone.items()}
            )
        if not redundant:
            self.tmp_out[index] = out.clone()
            
        return out
    
    def redundant_backward(self, index):
        if self.redundant_virtual_rank == self.pipeline_virtual_gpus - 1:
            return
        
        self.redundant_cached_output_micro_batches[index].backward(gradient=self.input_micro_batches[index].grad)

class RedundantVirtualAsync:
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

        self.virtual_gpus = [VirtualGPU(args, config, i, device, self.micro_batch_num, _StageFirst, _StageLast, _StageMiddle) for i in range(args.virtual_gpus * self.pp_rank, args.virtual_gpus * (self.pp_rank + 1))]
        self.model_recv_ready_event = {key:torch.cuda.Event(enable_timing=False, blocking=False) for key in self.virtual_gpus[-1].model.state_dict()}

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
            redundant=3
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
            self.schedulers = [get_linear_schedule_with_warmup(tmp_optimizers[i], args.warmup_steps, args.total_steps, ) for i in range(args.virtual_gpus)]
            if self.pp_rank != 0:
                redundant_tmp_optimizers = [create_optimizer(self.virtual_gpus[i].redundant_model, learning_rate=args.lr, optim=args.optimizer) for i in range(args.virtual_gpus)]
                self.redundant_optimizers = [get_fp16_optimizer(args, redundant_tmp_optimizers[i], device) for i in range(args.virtual_gpus)]
                self.redundant_schedulers = [get_linear_schedule_with_warmup(redundant_tmp_optimizers[i], args.warmup_steps, args.total_steps, ) for i in range(args.virtual_gpus)]
            else:
                redundant_tmp_optimizers = [create_optimizer(self.virtual_gpus[i].redundant_model, learning_rate=args.lr, optim=args.optimizer) for i in range(1, args.virtual_gpus)]
                self.redundant_optimizers = [get_fp16_optimizer(args, redundant_tmp_optimizers[i], device) for i in range(args.virtual_gpus - 1)]
                self.redundant_schedulers = [get_linear_schedule_with_warmup(redundant_tmp_optimizers[i], args.warmup_steps, args.total_steps, ) for i in range(args.virtual_gpus - 1)]
        else:
            self.optimizers = [create_optimizer(self.virtual_gpus[i].model, learning_rate=args.lr, optim=args.optimizer) for i in range(args.virtual_gpus)]
            self.schedulers = [get_linear_schedule_with_warmup(self.optimizers[i], args.warmup_steps, args.total_steps, ) for i in range(args.virtual_gpus)]
            if self.pp_rank != 0:
                self.redundant_optimizers = [create_optimizer(self.virtual_gpus[i].redundant_model, learning_rate=args.lr, optim=args.optimizer) for i in range(args.virtual_gpus)]
                self.redundant_schedulers = [get_linear_schedule_with_warmup(self.redundant_optimizers[i], args.warmup_steps, args.total_steps, ) for i in range(args.virtual_gpus)]
            else:
                self.redundant_optimizers = [create_optimizer(self.virtual_gpus[i].redundant_model, learning_rate=args.lr, optim=args.optimizer) for i in range(1, args.virtual_gpus)]
                self.redundant_schedulers = [get_linear_schedule_with_warmup(self.redundant_optimizers[i], args.warmup_steps, args.total_steps, ) for i in range(args.virtual_gpus - 1)]

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
        return self.invalid_times / self.total_times

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

    def forward_attack(self, input: torch.Tensor, last_input: torch.Tensor, last2output: torch.Tensor, index):
        if self.virtual_gpus[0].model.training:
            input_perturbation = torch.normal(mean=float(input.mean()), std=float(input.std()), size=tuple(input.shape), device=input.device)
            input.data.add_(input_perturbation)

            last_input_min = last_input.min()
            last_input_max = last_input.max()
            last_input_perturbation = torch.normal(mean=float(last_input.float().mean()), std=float(last_input.float().std()), size=tuple(last_input.shape), device=last_input.device)
            if last_input.dtype == torch.int64:
                last_input_perturbation = last_input_perturbation.round().to(torch.long)
            last_input.data.add_(last_input_perturbation)
            last_input.data.clamp_(last_input_min, last_input_max)

            last2output_perturbation = torch.normal(mean=float(last2output.mean()), std=float(last2output.std()), size=tuple(last2output.shape), device=last2output.device)
            last2output.data.add_(last2output_perturbation)

            self.attack_stage[index] = 1
        return input, last_input, last2output

    def virtual_forward(self, aux_input_data, index, input_ids_micro_batch=None, last_input=None, last2output=None):
        # last_input: last stage'input
        # last_output: last stage'output
        # last2output: equal last_input

        for i in range(len(self.virtual_gpus)):
            if i == 0:
                if self.virtual_gpus[i].virtual_rank == 0:
                    tmp_output = self.virtual_gpus[i].forward(self.input_micro_batches[index], aux_input_data, index, input_ids_micro_batch)
                    last_input = self.input_micro_batches[index].clone()
                elif self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1:
                    if last_input is not None:
                        valid, last_input = self.virtual_gpus[i].valid(last_input, last2output, self.input_micro_batches[index].clone(), aux_input_data, index)
                        if not valid:
                            return torch.zeros_like(self.input_micro_batches[index], device=self.device), torch.zeros_like(self.input_micro_batches[index], device=self.device), torch.zeros_like(self.input_micro_batches[index], device=self.device)
                    tmp_output = self.virtual_gpus[i].forward(self.input_micro_batches[index], aux_input_data, index, input_ids_micro_batch)
                else:
                    if last_input is not None:
                        valid, last_input = self.virtual_gpus[i].valid(last_input, last2output, self.input_micro_batches[index].clone(), aux_input_data, index)
                        if not valid:
                            return torch.zeros_like(self.input_micro_batches[index], device=self.device), torch.zeros_like(self.input_micro_batches[index], device=self.device), torch.zeros_like(self.input_micro_batches[index], device=self.device)
                    if self.mode == "train":
                        self.input_micro_batches[index].retain_grad()
                    tmp_output = self.virtual_gpus[i].forward(self.input_micro_batches[index], aux_input_data, index, input_ids_micro_batch)
            elif i == 1:
                if self.mode == "train" and i == len(self.virtual_gpus) - 1:
                    tmp_output.retain_grad()
                valid, last_input = self.virtual_gpus[i].valid(last_input, self.input_micro_batches[index], tmp_output.clone(), aux_input_data, index)
                if not valid:
                    return torch.zeros(tmp_output.size(), device=self.device), torch.zeros(tmp_output.size(), device=self.device), torch.zeros(tmp_output.size(), device=self.device)
                tmp_output = self.virtual_gpus[i].forward(tmp_output, aux_input_data, index, input_ids_micro_batch)
            else:
                if self.mode == "train" and i == len(self.virtual_gpus) - 1:
                    tmp_output.retain_grad()
                valid, last_input = self.virtual_gpus[i].valid(last_input, self.virtual_gpus[i - 2].tmp_out[index], tmp_output.clone(), aux_input_data, index)
                if not valid:
                    return torch.zeros(tmp_output.size(), device=self.device), torch.zeros(tmp_output.size(), device=self.device), torch.zeros(tmp_output.size(), device=self.device)
                tmp_output = self.virtual_gpus[i].forward(tmp_output, aux_input_data, index, input_ids_micro_batch)
            if self.virtual_gpus[i].virtual_rank == self.malicious_stage:
                tmp_output, last_input, self.virtual_gpus[i].tmp_out[index] = self.forward_attack(tmp_output, last_input, self.virtual_gpus[i].tmp_out[index], index)

        return tmp_output, last_input, self.virtual_gpus[-2].tmp_out[index]
    
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
                    current_micro_output, last_input, last2output = self.virtual_forward(aux_input_data, i)
                    if current_micro_output.sum() == 0:
                        self.success_stage[i] = 0
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    send_data = torch.stack((current_micro_output.data, last_input.data, last2output.data), dim=0)
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
                    last2output = _data[2].clone()
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    if _data.sum() != 0:
                        current_micro_output, last_input, last2output = self.virtual_forward(aux_input_data, i, input_ids_micro_batches[i], last_input, last2output)
                    else:
                        current_micro_output, last_input, last2output = _data[0].clone(), _data[1].clone(), _data[2].clone()
                    if current_micro_output.sum() == 0:
                        self.success_stage[i] = 0
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.forward_compressor.recv_decompress(
                        i, comm=self.comm, src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.input_micro_batches[i].data.copy_(_data[0])
                    last_input = _data[1].clone()
                    last2output = _data[2].clone()
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    if _data.sum() != 0:
                        current_micro_output, last_input, last2output = self.virtual_forward(aux_input_data, i, last_input=last_input, last2output=last2output)
                    else:
                        current_micro_output, last_input, last2output = _data[0].clone(), _data[1].clone(), _data[2].clone()
                    if current_micro_output.sum() == 0:
                        self.success_stage[i] = 0
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    send_data = torch.stack((current_micro_output.data, last_input.data, last2output.data), dim=0)
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
                    self.virtual_gpus[-1].redundant_backward(i)
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
                    self.virtual_gpus[-1].redundant_backward(i)
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
                    self.virtual_gpus[-1].redundant_backward(i)
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
                self.torch_recv_stream.record_event(self.status_recv_ready_event)
            with torch.cuda.stream(self.torch_send_stream):
                self.torch_send_stream.wait_event(self.status_recv_ready_event)
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.comm.send(self.malicious_stage, self.post_node_rank, cupy_send_stream)

    
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

    def get_redundant_models(self, model_name, virtual_num_layers):
        for gpu in self.virtual_gpus:
            if gpu.redundant_virtual_rank == self.pipeline_virtual_gpus - 1:
                continue
            if gpu.redundant_virtual_rank == 0:
                gpu.redundant_model.model[0].load_state_dict(
                    torch.load(f'{model_name}/pytorch_embs.pt')
                )
                for i in range(len(gpu.redundant_model.model) - 1):
                    gpu.redundant_model.model[i + 1].load_state_dict(
                        torch.load(f'{model_name}/pytorch_{i}.pt')
                    )
            else:
                _i = gpu.redundant_virtual_rank * virtual_num_layers
                for i in range(len(gpu.redundant_model.model)):
                    gpu.redundant_model.model[i].load_state_dict(
                        torch.load(f'{model_name}/pytorch_{i + _i}.pt')
                    )

    def optimizer_step(self):
        for i in range(len(self.virtual_gpus)):
            torch.nn.utils.clip_grad_norm_(self.virtual_gpus[i].model.parameters(), 1.0)
            if hasattr(self.virtual_gpus[i], "redundant_model"):
                torch.nn.utils.clip_grad_norm_(self.virtual_gpus[i].redundant_model.parameters(), 1.0)

        with torch.cuda.stream(self.torch_comp_stream):
            for i in range(len(self.optimizers)):
                self.optimizers[i].step()
                self.schedulers[i].step()
            for i in range(len(self.redundant_optimizers)):
                self.redundant_optimizers[i].step()
                self.redundant_schedulers[i].step()

    def sgd_iter(self, input_=None, target=None, sample_ids=None, 
                 aux_input_data=None, loss_func=torch.nn.functional.cross_entropy,
                 iter=0):
        self.comm.barrier()
        start_time = time.time()
        self.zero_input_grad()
        for i in range(len(self.optimizers)):
            self.optimizers[i].zero_grad(set_to_none=False)
        for i in range(len(self.redundant_optimizers)):
            self.redundant_optimizers[i].zero_grad(set_to_none=False)

        for step in range(self.gradient_accumulate_step):
            self.success_stage = torch.ones(self.micro_batch_num, dtype=torch.int, device=self.device)
            self.attack_stage = torch.zeros(self.micro_batch_num, dtype=torch.int, device=self.device)
            self.malicious_stage = torch.tensor(-1, dtype=torch.int, device=self.device)
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
        self.total_times += len(self.success_stage)
        self.invalid_times += torch.sum(self.success_stage == 0).item()
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
                    self.virtual_gpus[i].redundant_model.train()

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
                    current_micro_output, _, _ = self.virtual_forward(aux_input_data, i)
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
                    current_micro_output, _, _ = self.virtual_forward(aux_input_data, i)
                    current_micro_output = pred_func(current_micro_output, labels[i])
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    current_micro_output, _, _ = self.virtual_forward(aux_input_data, i)
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
        


