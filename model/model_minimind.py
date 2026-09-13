import math,torch,torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PretrainedConfig,PreTrainedModel,GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.modeling_tf_utils import init_copy_embeddings


#模型配置
class MiniMindConfig(PretrainedConfig):
    def __init__(self, hidden_size=768,num_hidden_layers=8,use_moe=False,**kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

#模型组装
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float = 1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def norm(self,x):
        return x * torch.rsqrt(torch.mean(x**2,dim=-1,keepdim=True) + self.eps)

    def forward(self,x):
        return (self.gamma * self.norm(x.float())).type_as(x)

def precomputer_freqs_cis(dim:int,end:int=int(32*1024),rope_base:float=1e6,rope_scaling:dict=None):
    freqs,attn_factor = 1.0/(rope_base ** (torch.arange(0,dim,2)[: (dim // 2)].float() / dim )),1.0
    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow,attn_factor = (
            rope_scaling.get("original_max_position_embeddings",2048),
            rope_scaling.get("factor",16),
            rope_scaling.get("beta_fast",32.),
            rope_scaling.get("beta_slow",1.),
            rope_scaling.get("attn_factor",1.)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b:(dim * math.log(orig_max / (b*2*math.pi)) / 2*(math.log(rope_base)))
            low,high = max(math.floor(inv_dim(beta_fast)),0),min(math.ceil(inv_dim(beta_slow)),dim//2 -1)
            ramp = torch.clamp((torch.arange(dim//2,device=freqs.device).float()),0,1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end,device=freqs.device)
    freqs = torch.outer(t,freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1) * attn_factor
    return freqs_cos,freqs_sin

def apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim = 1):
    def rotate_half(x):
        return torch.cat((-x[...,x.shape[-1] // 2 :],x[...,:x.shape[-1] // 2]),dim=-1)
    q_embed = ((q*cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q)*sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k*cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k)*sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed,k_embed

def repeat_kv(x:torch.Tensor,n_rep:int) -> torch.Tensor:
    bs,slen,num_key_value_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:,:,:,None,:].expand(bs,slen,num_key_value_heads,n_rep,head_dim).reshape(bs,slen,num_key_value_heads * n_rep,head_dim))


class Attention(nn.Module):
    def __init__(self,config : MiniMindConfig):
        super(Attention, self).__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size,config.num_attention_heads * self.head_dim,bias=False)
        self.k_proj = nn.Linear(config.hidden_size,config.num_key_value_heads * self.head_dim,bias=False)
        self.v_proj = nn.Linear(config.hidden_size,config.num_key_value_heads * self.head_dim,bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim,config.hidden_size,bias=False)
        self.q_norm = RMSNorm(config.hidden_size,eps = config.rms_norm_eps)
        self.k_norm = RMSNorm(config.hidden_size,eps = config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
