import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Callable
from transformers.utils import logging
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


from .compression import (
    RKV,
    SnapKV,
    StreamingLLM,
    H2O,
)

import math
import torch.nn.functional as F

KV_COMPRESSION_MAP = {
    "rkv": RKV,
    "snapkv": SnapKV,
    "streamingllm": StreamingLLM,
    "h2o": H2O,
}

logger = logging.get_logger(__name__)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def LlamaAttention_init(
    self, config: LlamaConfig, layer_idx: int, compression_config: dict
):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.num_key_value_heads = config.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta

    self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    self.q_proj = nn.Linear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.k_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        layer_idx=self.layer_idx, model_config=self.config, model_type="llama3", **compression_config["method_config"] 
    )
    # =============== New logic end =================

def LlamaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    kvcomm_should_compress = kwargs.pop("kvcomm_should_compress", None)
    kvcomm_state_key = kwargs.pop("kvcomm_state_key", None)
    if kvcomm_should_compress is None:
        kvcomm_should_compress = self.config.compression

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        cached_queries = past_key_value.query_cache[self.layer_idx]
        if kvcomm_should_compress is None:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
                state_key=kvcomm_state_key,
            )

            if self.config.update_kv is True:
                past_key_value.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,
                    cache_kwargs,
                )
            else:
                past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

        elif kvcomm_should_compress is True:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
                state_key=kvcomm_state_key,
            )

            if self.config.update_kv is True:
                past_key_value.key_cache[self.layer_idx] = key_states_compress
                past_key_value.value_cache[self.layer_idx] = value_states_compress
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        # =============== decoding-time compression end ===============

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def CausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    kvcomm_state_key = kwargs.pop("kvcomm_state_key", None)
    kwargs.pop("kvcomm_should_compress", None)

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if not hasattr(self, "_kvcomm_seq_states"):
        self._kvcomm_seq_states = {}
    state_key = kvcomm_state_key or "__default__"
    if state_key not in self._kvcomm_seq_states:
        self._kvcomm_seq_states[state_key] = {
            "length": 0,
            "after_think": False,
            "next_compression": None,
        }
    seq_state = self._kvcomm_seq_states[state_key]

    input_len = 0
    if input_ids is not None:
        input_len = input_ids.shape[1]
    elif inputs_embeds is not None:
        input_len = inputs_embeds.shape[1]

    is_prefill = False
    if past_key_values is None:
        is_prefill = True
    else:
        try:
            is_prefill = len(past_key_values) == 0
        except TypeError:
            is_prefill = False

    if is_prefill:
        seq_state["length"] = input_len
        if self.config.compression_content == "think":
            seq_state["after_think"] = False
        seq_state["next_compression"] = None
    else:
        seq_state["length"] += input_len

    kvcomm_should_compress = seq_state.get("next_compression", None)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        kvcomm_state_key=state_key,
        kvcomm_should_compress=kvcomm_should_compress,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    # =============== Step-level Compression logic start ===============
    # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
    predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

    if self.config.compression_content == "think" and seq_state["after_think"] is False:
        seq_state["after_think"] = (
            predicted_token_ids[0].cpu().item() in self.after_think_token_ids
        )

    if self.config.divide_method == "newline":
        is_newline = predicted_token_ids[0].cpu().item() in self.newline_token_ids
    elif self.config.divide_method == "step_length":
        is_newline = seq_state["length"] % self.config.divide_length == 0
    else:
        raise ValueError(f"Invalid divide_method: {self.config.divide_method}")

    if self.config.compression_content == "think" and seq_state["after_think"] is True:
        is_newline = False

    seq_state["next_compression"] = is_newline
    # =============== Step-level Compression logic end =================

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
