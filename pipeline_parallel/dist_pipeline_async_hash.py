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

class SHA256:
    def __init__(self):
        self.constants = (
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2)
        
        self.h = (
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19)

    def rightrotate(self, x, b):
        return ((x >> b) | (x << (32 - b))) & ((2**32)-1)

    def Pad(self, W):
        return bytes(W, "ascii") + b"\x80" + (b"\x00" * ((55 if (len(W) % 64) < 56 else 119) - (len(W) % 64))) + (
            (len(W) << 3).to_bytes(8, "big"))

    def Compress(self, Wt, Kt, A, B, C, D, E, F, G, H):
        return ((H + (self.rightrotate(E, 6) ^ self.rightrotate(E, 11) ^ self.rightrotate(E, 25)) + (
                    (E & F) ^ (~E & G)) + Wt + Kt) + (
                            self.rightrotate(A, 2) ^ self.rightrotate(A, 13) ^ self.rightrotate(A, 22)) + (
                            (A & B) ^ (A & C) ^ (B & C))) & ((2**32)-1), A, B, C, (D + (
                    H + (self.rightrotate(E, 6) ^ self.rightrotate(E, 11) ^ self.rightrotate(E, 25)) + (
                        (E & F) ^ (~E & G)) + Wt + Kt)) & ((2**32)-1), E, F, G

    def hash(self, message):
        message = self.Pad(message)
        digest = list(self.h)

        for i in range(0, len(message), 64):
            S = message[i: i + 64]
            W = [int.from_bytes(S[e: e + 4], "big") for e in range(0, 64, 4)] + ([0] * 48)

            #构造64个word
            for j in range(16, 64):
                W[j] = (W[j - 16] + (
                            self.rightrotate(W[j - 15], 7) ^ self.rightrotate(W[j - 15], 18) ^ (W[j - 15] >> 3)) + W[
                            j - 7] + (self.rightrotate(W[j - 2], 17) ^ self.rightrotate(W[j - 2], 19) ^ (
                            W[j - 2] >> 10))) & ((2**32)-1)

            A, B, C, D, E, F, G, H = digest

            for j in range(64):
                A, B, C, D, E, F, G, H = self.Compress(W[j], self.constants[j], A, B, C, D, E, F, G, H)

        return sum([A, B, C, D, E, F, G, H])
        # return "".join(format(h, "02x") for h in b"".join(
        #     d.to_bytes(4, "big") for d in [(x + y) & ((2**32)-1) for x, y in zip(digest, (A, B, C, D, E, F, G, H))]))

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
        self.micro_batch_size = args.micro_batch_size
        self.top_n = args.top_n
        self.history = []
        self.distance_mode = args.distance
        self.seq_length = args.seq_length
        self.embedding_dim = args.embedding_dim
        self.encoder = SHA256()

        self.device = device

        if virtual_rank == 0:
            self.model = _StageFirst(args, config, device)
        elif virtual_rank == self.pipeline_virtual_gpus - 1:
            self.model = _StageLast(args, config, device)
        else:
            self.model = _StageMiddle(args, config, device)

        # self.valid_model = torch.nn.Conv1d(self.seq_length, self.seq_length, 1, bias=False, device=device)

        if self.use_fp16:
            self.model.half()
            # self.valid_model.half()

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
    
    def get_hash_code(self, input: torch.Tensor):
        # return self.valid_model(input)
        row_sum = torch.sum(input, dim=-1).view(self.micro_batch_size, -1, 1)
        # hash_code = torch.zeros_like(row_sum, device=self.device, dtype=self.dtype)
        # for i in range(len(row_sum)):
        #     for j in range(len(row_sum[i])):
        #         hash_code[i][j][0] = self.encoder.hash(str(row_sum[i][j][0].item()))
        return row_sum

    def valid(self, _data, hash_code):
        return torch.equal(self.get_hash_code(_data), hash_code)
        

class VirtualHashAsync:
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
        self.valid_model_recv_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
        
        
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

        # self.get_valid_model()
        # torch.cuda.synchronize()
        # self.comm.barrier()

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
            hash=True
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

    def get_valid_model(self):
        valid_model_dict = self.virtual_gpus[0].valid_model.state_dict()
        if self.pp_rank == 0:
            for i in range(len(self.virtual_gpus)):
                self.virtual_gpus[i].valid_model.load_state_dict(valid_model_dict)
            with torch.cuda.stream(self.torch_send_stream):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.comm.send(valid_model_dict["weight"], self.post_node_rank, cupy_send_stream)
        elif self.pp_rank == self.pipeline_group_size - 1:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.comm.recv(valid_model_dict["weight"], self.pre_node_rank, cupy_recv_stream)
                for i in range(len(self.virtual_gpus)):
                    self.virtual_gpus[i].valid_model.load_state_dict(valid_model_dict)
        else:
            with torch.cuda.stream(self.torch_recv_stream):
                cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                self.comm.recv(valid_model_dict["weight"], self.pre_node_rank, cupy_recv_stream)
                self.torch_recv_stream.record_event(self.valid_model_recv_ready_event)
                for i in range(len(self.virtual_gpus)):
                    self.virtual_gpus[i].valid_model.load_state_dict(valid_model_dict)
            with torch.cuda.stream(self.torch_send_stream):
                cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                self.torch_send_stream.wait_event(self.valid_model_recv_ready_event)
                self.comm.send(valid_model_dict["weight"], self.post_node_rank, cupy_send_stream)

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

    def forward_attack(self, input: torch.Tensor, hash_code: torch.Tensor, index):
        p = random.random()
        if self.virtual_gpus[0].model.training and p < self.forward_attack_rate:
            input_perturbation = torch.normal(mean=float(input.mean()), std=float(input.std()), size=tuple(input.shape), device=input.device)
            input.data.add_(input_perturbation)
            hash_code_perturbation = torch.normal(mean=float(hash_code.float().mean()), std=float(hash_code.float().std()), size=tuple(hash_code.shape), device=hash_code.device)
            hash_code.data.add_(hash_code_perturbation)
            self.attack_stage[index] = 1
        return input, hash_code

    def virtual_forward(self, aux_input_data, index, input_ids_micro_batch=None, hash_code=None):
        for i in range(len(self.virtual_gpus)):
            if i == 0:
                if self.virtual_gpus[i].virtual_rank == 0:
                    tmp_output = self.virtual_gpus[i].forward(self.input_micro_batches[index], aux_input_data, index, input_ids_micro_batch)
                    with torch.no_grad():
                        hash_code = self.virtual_gpus[i].get_hash_code(tmp_output)
                    tmp_output, hash_code = self.forward_attack(tmp_output, hash_code, index)
                elif self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1:
                    with torch.no_grad():
                        if hash_code is not None and not self.virtual_gpus[i].valid(self.input_micro_batches[index], hash_code):
                            return torch.zeros_like(self.input_micro_batches[index], device=self.device), torch.zeros_like(hash_code, device=self.device)
                    tmp_output = self.virtual_gpus[i].forward(self.input_micro_batches[index], aux_input_data, index, input_ids_micro_batch)
                    with torch.no_grad():
                        hash_code = self.virtual_gpus[i].get_hash_code(tmp_output)
                    tmp_output, hash_code = self.forward_attack(tmp_output, hash_code, index)
                else:
                    with torch.no_grad():
                        if hash_code is not None and not self.virtual_gpus[i].valid(self.input_micro_batches[index], hash_code):
                            return torch.zeros_like(self.input_micro_batches[index], device=self.device), torch.zeros_like(hash_code, device=self.device)
                    tmp_output = self.virtual_gpus[i].forward(self.input_micro_batches[index], aux_input_data, index, input_ids_micro_batch)
            else:
                if self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1:
                    with torch.no_grad():
                        if not self.virtual_gpus[i].valid(tmp_output, hash_code):
                            return torch.zeros_like(tmp_output, device=self.device), torch.zeros_like(hash_code, device=self.device)
                    tmp_output = self.virtual_gpus[i].forward(tmp_output, aux_input_data, index, input_ids_micro_batch)
                    with torch.no_grad():
                        hash_code = self.virtual_gpus[i].get_hash_code(tmp_output)
                    tmp_output, hash_code = self.forward_attack(tmp_output, hash_code, index)
                else:
                    with torch.no_grad():
                        if not self.virtual_gpus[i].valid(tmp_output, hash_code):
                            return torch.zeros_like(tmp_output, device=self.device), torch.zeros_like(hash_code, device=self.device)
                    tmp_output = self.virtual_gpus[i].forward(tmp_output, aux_input_data, index, input_ids_micro_batch)

            # if self.virtual_gpus[i].model.training and self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1:
            #     tmp_output.register_hook(attack_backward_hook(self.backward_attack_rate))
            # if self.virtual_gpus[i].virtual_rank != self.pipeline_virtual_gpus - 1 and (self.virtual_gpus[i].virtual_rank + 1) % len(self.virtual_gpus) == 0:
            #     tmp_output.register_hook(print_tensor_gradient)
        return tmp_output, hash_code
    
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
                    current_micro_output, hash_code = self.virtual_forward(aux_input_data, i)
                    if torch.all(current_micro_output == 0):
                        self.success_stage[i] = 0
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    # send_data = torch.stack((current_micro_output.data, hash_code.data), dim=0)
                    send_data = torch.cat((current_micro_output.data, hash_code.data), dim=-1)
                    self.forward_compressor.compress_send(
                        send_data.data, i_micro_batch=i,
                        comm=self.comm, dst=self.post_node_rank, stream=cupy_send_stream
                    )
            elif self.pp_rank == self.pipeline_group_size - 1:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.forward_compressor.recv_decompress(
                        i, comm=self.comm, src=self.pre_node_rank, stream=cupy_recv_stream)
                    self.input_micro_batches[i].data.copy_(_data[:, :, :self.embedding_dim])
                    hash_code = _data[:, :, self.embedding_dim:]
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    if torch.any(_data):
                        current_micro_output, hash_code = self.virtual_forward(aux_input_data, i, input_ids_micro_batches[i], hash_code)
                    else:
                        current_micro_output, hash_code = _data[:, :, :self.embedding_dim].clone(), _data[:, :, self.embedding_dim:].clone()
                    if torch.all(current_micro_output == 0):
                        self.success_stage[i] = 0
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
            else:
                with torch.cuda.stream(self.torch_recv_stream):
                    cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                    _data = self.forward_compressor.recv_decompress(
                        i, comm=self.comm, src=self.pre_node_rank, stream=cupy_recv_stream)
                    # self.input_micro_batches[i].data.copy_(_data[0])
                    # hash_code = _data[1].clone()
                    self.input_micro_batches[i].data.copy_(_data[:, :, :self.embedding_dim])
                    hash_code = _data[:, :, self.embedding_dim:]
                    self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                with torch.cuda.stream(self.torch_comp_stream):
                    self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                    if torch.any(_data):
                        current_micro_output, hash_code = self.virtual_forward(aux_input_data, i, hash_code=hash_code)
                    else:
                        # current_micro_output, hash_code = _data[0].clone(), _data[1].clone()
                        current_micro_output, hash_code = _data[:, :, :self.embedding_dim].clone(), _data[:, :, self.embedding_dim:].clone()
                    if torch.all(current_micro_output == 0):
                        self.success_stage[i] = 0
                    self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                with torch.cuda.stream(self.torch_send_stream):
                    cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                    self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                    # send_data = torch.stack((current_micro_output.data, hash_code.data), dim=0)
                    send_data = torch.cat((current_micro_output.data, hash_code.data), dim=-1)
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
        

