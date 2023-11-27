from torch import nn
from .t5_module import EncDecEmbeddings, EncBlock, DecBlock, EncHead, DecHead


class T5StageBase(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self._to_cpu = False # (args.dist_backend == "gloo")
        self.config = config
        self.pipeline_virtual_gpus = args.pipeline_virtual_gpus
        self.model_name = args.model_name
        self.virtual_num_layers = args.virtual_num_layers * 2

    def _create_first_layer(self):
        return EncDecEmbeddings.from_pretrained(self.model_name, self.config)

    def _create_dec_head_layer(self):
        return DecHead.from_pretrained(self.model_name, self.config)
    
    def _create_enc_head_layer(self):
        print("_create_enc_head_layer")
        return EncHead.from_pretrained(self.model_name, self.config)

    def _create_dec_layer(self, index):
        return DecBlock.from_pretrained(self.model_name, self.config, index)
    
    def _create_enc_layer(self, index):
        return EncBlock.from_pretrained(self.model_name, self.config, index)


class T5StageFirst(T5StageBase):
    def __init__(self, args, config, index, device):
        super().__init__(args, config, index)
        self.device = device
        module_list = [self._create_first_layer()]
        for i in range(self.virtual_num_layers):
            print("layer: ", i)
            if i + index * self.virtual_num_layers < config.num_layers:
                module_list.append(self._create_enc_layer(i + index * self.virtual_num_layers))
            else:
                module_list.append(self._create_dec_layer(i + index * self.virtual_num_layers - config.num_layers))
            if i + index * self.virtual_num_layers == config.num_layers - 1:
                module_list.append(self._create_enc_head_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device))
        return out.cpu() if self._to_cpu else out


class T5StageMiddle(T5StageBase):
    def __init__(self, args, config, index, device):
        super().__init__(args, config, index)
        self.device = device
        module_list = []
        for i in range(self.virtual_num_layers):
            print("layer: ", i + index * self.virtual_num_layers)
            if i + index * self.virtual_num_layers < config.num_layers:
                module_list.append(self._create_enc_layer(i + index * self.virtual_num_layers))
            else:
                module_list.append(self._create_dec_layer(i + index * self.virtual_num_layers - config.num_layers))
            if i + index * self.virtual_num_layers == config.num_layers - 1:
                module_list.append(self._create_enc_head_layer())
            
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out


class T5StageLast(T5StageBase):
    def __init__(self, args, config, index, device):
        super().__init__(args, config, index)
        self.device = device
        module_list = []
        for i in range(self.virtual_num_layers):
            print("layer: ", i + index * self.virtual_num_layers)
            if i + index * self.virtual_num_layers < config.num_layers:
                module_list.append(self._create_enc_layer(i + index * self.virtual_num_layers))
            else:
                module_list.append(self._create_dec_layer(i + index * self.virtual_num_layers - config.num_layers))
            if i + index * self.virtual_num_layers == config.num_layers - 1:
                module_list.append(self._create_enc_head_layer())
        module_list.extend(self._create_dec_head_layer())
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