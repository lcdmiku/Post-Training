import torch
from torch import nn

class LoRA(nn.Module):
    def __init__(self,in_features,out_features,rank=16):
        super(LoRA,self).__init__()
        self.rank = rank
        self.A = nn.Linear(in_features,self.rank,bias=False)
        self.B = nn.Linear(self.rank,out_features,bias=False)
        nn.init.kaiming_uniform_(self.A.weight)
        #B初始化为0，保证一开始与原模型映射相同
        nn.init.zeros_(self.B.weight)

    def forward(self,x):
        return self.B(self.A(x))

def apply_lora(model,rank=16):
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features,module.out_features,rank).to(model.device)
            setattr(module,'lora',lora)
            origin_forward = module.forward
            def new_forward(x,layer1=origin_forward,layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = new_forward

def load_lora(model,path):
    state_dict = torch.load(path,map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    for name,module in model.named_modules():
        if hasattr(module,'lora'):
            lora_state = {(k.replace(f'{name}.lora.',"")): v for k,v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)

def save_lora(model,path):
    state_dict = {}
    raw_model = getattr(model,'_orig_mod',model)
    for name,module in raw_model.named_modules():
        if hasattr(module,'lora'):
            clean_name = name[7:] if name.startswith('module.') else name
            lora_state = {f'{clean_name}.lora.{k}' : v.cpu().half() for k,v in module.lora.state_dict().items()}
            state_dict.update(lora_state)

    torch.save(state_dict,path)


def merge_lora(model,load_path,save_path):
    load_lora(model,load_path)
    raw_model = getattr(model,'_orig_mod',model)
    state_dict = {k:v.cpu().half() for k,v in raw_model.state_dict().items() if '.lora.' not in k}
    for name,module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)