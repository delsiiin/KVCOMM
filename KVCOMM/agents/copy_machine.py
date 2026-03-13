from typing import Any,Dict
import asyncio

from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.llm.config import KVCommConfig

@AgentRegistry.register('CopyMachine')
class CopyMachine(Node):
    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
        compress_mode: bool = False,
        compress_method: str = "rkv",
        compress_budget: int = 1024,
        compress_divide_length: int = 128,
        attn_heatmap_mode: bool = False,
        attn_heatmap_layer: int | None = None,
        attn_heatmap_output_dir: str | None = None,
        attn_heatmap_run_tag: str | None = None,
        model_dtype: str = "float16",
    ):
        super().__init__(id, "CopyMachine" ,domain, llm_name)
        prefix = ""

        self.llm = LLMRegistry.get(
            llm_name,
            prefix=prefix,
            llm_config=llm_config,
            compress_mode=compress_mode,
            compress_method=compress_method,
            compress_budget=compress_budget,
            compress_divide_length=compress_divide_length,
            attn_heatmap_mode=attn_heatmap_mode,
            attn_heatmap_layer=attn_heatmap_layer,
            attn_heatmap_output_dir=attn_heatmap_output_dir,
            attn_heatmap_run_tag=attn_heatmap_run_tag,
            model_dtype=model_dtype,
        )
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.llm.set_id(self.id, self.role)
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Dict],
        temporal_info:Dict[str,Dict],
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare prompts in default mode."""
        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n"
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if len(temporal_str) else ""
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

        inputs = asyncio.run(
            self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                **kwargs,
            )
        )
        system_prompt = inputs["system_prompt"]
        user_prompt = inputs["user_prompt"]
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs):
        """Handle asynchronous execution in default mode."""
        request_uid = input.get("_request_uid")
        inputs = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            **kwargs,
        )
        system_prompt = inputs["system_prompt"]
        user_prompt = inputs["user_prompt"]
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        result = await self.llm.agen(
            message,
            max_tokens=self.llm.DEFAULT_MAX_TOKENS,
            request_uid=request_uid,
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
            round_index=input.get("_round_index"),
        )
        return result
