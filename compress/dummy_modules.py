import torch
import torch.nn as nn
import torch.nn.functional as F
# import copy
# import cupy
from utils.common_utils import print_tensor
from communication.comm_utils import get_pipeline_parallel_rank


class NoCompression:
    def __init__(self, *args, **kargs):
        pass
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=torch.float32, redundant=0, hash=False):
        if redundant:
            self.buffers = [
                torch.zeros((redundant, micro_batch_size, seq_length, embedding_dim), 
                            requires_grad=False, device=device, dtype=dtype,
                        ) for _ in range(batch_size//micro_batch_size)
            ]
        elif hash:
            self.buffers = [
                torch.zeros((micro_batch_size, seq_length, embedding_dim + 1), 
                            requires_grad=False, device=device, dtype=dtype,
                        ) for _ in range(batch_size//micro_batch_size)
            ]
        else:
            self.buffers = [
                torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                            requires_grad=False, device=device, dtype=dtype,
                        ) for _ in range(batch_size//micro_batch_size)
            ]
        
    def compress(self, x):
        return x
        
    def decompress(self, x):
        return x
        
    def compress_send(self, x, i_micro_batch, comm, dst, stream):
        # print_tensor(x, f"rank: {get_pipeline_parallel_rank()} batch: {i_micro_batch}")
        comm.send(x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        recv_buffer = self.buffers[i_micro_batch]
        comm.recv(recv_buffer, src=src, stream=stream)
        return recv_buffer
    