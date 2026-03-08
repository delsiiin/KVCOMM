from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from .modeling import (
    LlamaAttention_init,
    LlamaAttention_forward,
    CausalLM_forward,
)

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .flash_attn.flash_attention import flash_attention_forward

def replace_llama(compression_config):
    def init_wrapper(self, config, layer_idx):
        LlamaAttention_init(self, config, layer_idx, compression_config)

    modeling_llama.LlamaAttention.__init__ = init_wrapper
    modeling_llama.LlamaAttention.forward = LlamaAttention_forward
    modeling_llama.LlamaForCausalLM.forward = CausalLM_forward

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
