from torch import nn
from .hf_opt_module import GPTEmbeddings, GPTBlock, GPTLMHead


class OPTStageBase(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self._to_cpu = False # (args.dist_backend == "gloo")
        self.config = config
        self.virtual_num_layers = args.virtual_num_layers

    def _create_first_layer(self):
        return GPTEmbeddings(self.config)

    def _create_last_layer(self):
        return GPTLMHead(self.config)

    def _create_transformer_layer(self):
        return GPTBlock(self.config) # TODO: checkpoint


class OPTStageFirst(OPTStageBase):
    def __init__(self, args, config, device):
        super().__init__(args, config)
        self.device = device
        module_list = [self._create_first_layer()]
        for _ in range(self.virtual_num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device))
        return out.cpu() if self._to_cpu else out


class OPTStageMiddle(OPTStageBase):
    def __init__(self, args, config, device):
        super().__init__(args, config)
        self.device = device
        module_list = []
        for _ in range(self.virtual_num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out


class OPTStageLast(OPTStageBase):
    def __init__(self, args, config, device):
        super().__init__(args, config)
        self.device = device
        module_list = []
        for _ in range(self.virtual_num_layers):
            module_list.append(self._create_transformer_layer())
        module_list.append(self._create_last_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x, input_ids=None):
        if input_ids is None:
            out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        else:
            out = x.to(self.device) if self._to_cpu else x
            input_ids = input_ids.to(self.device) if self._to_cpu else input_ids
            for layer in self.model[:-1]:
                out = layer(out)

            out = self.model[-1](out, input_ids=input_ids)
            
        return out.cpu() if self._to_cpu else out