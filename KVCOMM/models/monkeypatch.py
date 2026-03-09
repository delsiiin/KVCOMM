from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from .modeling import (
    LlamaAttention_init,
    LlamaAttention_forward,
    Qwen2Attention_init,
    Qwen2Attention_forward,
    CausalLM_forward,
)

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .flash_attn.flash_attention import flash_attention_forward

def replace_llama(compression_config):
    def init_wrapper(self, config, layer_idx):
        LlamaAttention_init(self, config, layer_idx, compression_config)

    original_prepare_inputs_for_generation = getattr(
        modeling_llama.LlamaForCausalLM,
        "_kvcomm_original_prepare_inputs_for_generation",
        None,
    )
    if original_prepare_inputs_for_generation is None:
        original_prepare_inputs_for_generation = (
            modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
        )
        modeling_llama.LlamaForCausalLM._kvcomm_original_prepare_inputs_for_generation = (
            original_prepare_inputs_for_generation
        )

    def prepare_inputs_for_generation_wrapper(
        self,
        *args,
        kvcomm_state_key=None,
        **kwargs,
    ):
        model_inputs = original_prepare_inputs_for_generation(self, *args, **kwargs)
        if kvcomm_state_key is not None:
            model_inputs["kvcomm_state_key"] = kvcomm_state_key
        return model_inputs

    modeling_llama.LlamaAttention.__init__ = init_wrapper
    modeling_llama.LlamaAttention.forward = LlamaAttention_forward
    modeling_llama.LlamaForCausalLM.forward = CausalLM_forward
    modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = (
        prepare_inputs_for_generation_wrapper
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward


def replace_qwen2(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen2Attention_init(self, config, layer_idx, compression_config)

    original_prepare_inputs_for_generation = getattr(
        modeling_qwen2.Qwen2ForCausalLM,
        "_kvcomm_original_prepare_inputs_for_generation",
        None,
    )
    if original_prepare_inputs_for_generation is None:
        original_prepare_inputs_for_generation = (
            modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation
        )
        modeling_qwen2.Qwen2ForCausalLM._kvcomm_original_prepare_inputs_for_generation = (
            original_prepare_inputs_for_generation
        )

    def prepare_inputs_for_generation_wrapper(
        self,
        *args,
        kvcomm_state_key=None,
        **kwargs,
    ):
        model_inputs = original_prepare_inputs_for_generation(self, *args, **kwargs)
        if kvcomm_state_key is not None:
            model_inputs["kvcomm_state_key"] = kvcomm_state_key
        return model_inputs

    modeling_qwen2.Qwen2Attention.__init__ = init_wrapper
    modeling_qwen2.Qwen2Attention.forward = Qwen2Attention_forward
    modeling_qwen2.Qwen2ForCausalLM.forward = CausalLM_forward
    modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = (
        prepare_inputs_for_generation_wrapper
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
