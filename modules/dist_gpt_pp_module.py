from torch import nn
from .gpt_modules import GPTEmbeddings, GPTBlock, GPTClassificationHead, GPTLMHead
from utils.common_utils import print_tensor_gradient
from attack.attack import *


class GPTStageBase(nn.Module):
    def __init__(self, args, config):
        super(GPTStageBase, self).__init__()
        self._to_cpu = (args.dist_backend == "gloo")
#         self._vocab_size = vocab_size
        self._embedding_dim = args.embedding_dim  # embedding dimension
        self._seq_length = args.seq_length
#         self._num_classes = num_classes
        # the dimension of the feedforward aws_network model in nn.TransformerEncoder
        self._feedforward_dim = args.embedding_dim * 4
        self._num_heads = args.num_heads  # the number of heads in the multi-head attention models
        # self._num_layers = args.num_layers
        self._task_type = getattr(args, 'task_type', 'classification')
        
        self.config = config
        self.backward_attack_rate = args.backward_attack_rate
        self.virtual_num_layers = args.virtual_num_layers
        self.virtual_gpus = args.virtual_gpus

    def _create_first_layer(self):
        return GPTEmbeddings(self.config)

    def _create_last_layer(self):
        if self._task_type == 'classification':
            return GPTClassificationHead(self.config)
        elif self._task_type == 'language_model':
            return GPTLMHead(self.config)
        raise Exception('unknown data type')

    def _create_transformer_layer(self):
        return GPTBlock(self.config) # TODO: checkpoint


class GPTStageFirst(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageFirst, self).__init__(args, config)
        self.device = device
        module_list = [self._create_first_layer()]
        for _ in range(self.virtual_num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        # if not self.model.training:
        #     out = self.model(x.to(self.device))
        # else:
        #     out = self.model[0](x.to(self.device))
        #     for i in range(self.virtual_gpus):
        #         out = self.model[i * self.virtual_num_layers + 1:(i + 1) * self.virtual_num_layers + 1](out)
        #         out.register_hook(attack_backward_hook(self.backward_attack_rate))
        # out.register_hook(print_tensor_gradient)
        out = self.model(x.to(self.device))
        return out.cpu() if self._to_cpu else out


class GPTStageMiddle(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageMiddle, self).__init__(args, config)
        self.device = device
        module_list = []
        for _ in range(self.virtual_num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        # if not self.model.training:
        #     out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        # else:
        #     out = x.to(self.device) if self._to_cpu else x
        #     for i in range(self.virtual_gpus):
        #         out = self.model[i * self.virtual_num_layers:(i + 1) * self.virtual_num_layers](out)
        #         out.register_hook(attack_backward_hook(self.backward_attack_rate))
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out


class GPTStageLast(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageLast, self).__init__(args, config)
        self.device = device
        module_list = []
        for _ in range(self.virtual_num_layers):
            module_list.append(self._create_transformer_layer())
        module_list.append(self._create_last_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x, input_ids=None):
        # if input_ids is None:
        #     if not self.model.training:
        #         out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        #     else:
        #         out = x.to(self.device) if self._to_cpu else x
        #         for i in range(self.virtual_gpus):
        #             out = self.model[i * self.virtual_num_layers:(i + 1) * self.virtual_num_layers](out)
        #             out.register_hook(attack_backward_hook(self.backward_attack_rate))
        #         out = self.model[-1](out)
        # else:
        #     out = x.to(self.device) if self._to_cpu else x
        #     input_ids = input_ids.to(self.device) if self._to_cpu else input_ids
        #     if not self.model.training:
        #         for layer in self.model[:-1]:
        #             out = layer(out)
        #     else:
        #         for i in range(self.virtual_gpus):
        #             out = self.model[i * self.virtual_num_layers:(i + 1) * self.virtual_num_layers](out)
        #             if i != self.virtual_gpus - 1:
        #                 out.register_hook(attack_backward_hook(self.backward_attack_rate))
        #     out = self.model[-1](out, input_ids=input_ids)

        if input_ids is None:
            out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        else:
            out = x.to(self.device) if self._to_cpu else x
            input_ids = input_ids.to(self.device) if self._to_cpu else input_ids
            for layer in self.model[:-1]:
                out = layer(out)

            out = self.model[-1](out, input_ids=input_ids)
            
        return out.cpu() if self._to_cpu else out