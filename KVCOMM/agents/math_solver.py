import asyncio
from typing import Any,Dict
from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.tools.coding.python_executor import execute_code_get_return
from KVCOMM.llm.config import KVCommConfig

@AgentRegistry.register('MathSolver')
class MathSolver(Node):
    """Math agent that aggregates peer signals and computes final answers."""
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
        flowkv_mode: bool = False,
        flowkv_segment_granularity: str = "per_agent",
        flowkv_budget_bias: str = "history_first",
        flowkv_core_reserve: int = 128,
        flowkv_min_agent_budget: int = 32,
        attn_heatmap_mode: bool = False,
        attn_heatmap_layer: int | None = None,
        attn_heatmap_output_dir: str | None = None,
        attn_heatmap_run_tag: str | None = None,
        model_dtype: str = "float16",
    ):
        super().__init__(id, "MathSolver" ,domain, llm_name)
        prefix = "A: "
        self.llm = LLMRegistry.get(
            llm_name,
            prefix=prefix,
            llm_config=llm_config,
            compress_mode=compress_mode,
            compress_method=compress_method,
            compress_budget=compress_budget,
            compress_divide_length=compress_divide_length,
            flowkv_mode=flowkv_mode,
            flowkv_segment_granularity=flowkv_segment_granularity,
            flowkv_budget_bias=flowkv_budget_bias,
            flowkv_core_reserve=flowkv_core_reserve,
            flowkv_min_agent_budget=flowkv_min_agent_budget,
            attn_heatmap_mode=attn_heatmap_mode,
            attn_heatmap_layer=attn_heatmap_layer,
            attn_heatmap_output_dir=attn_heatmap_output_dir,
            attn_heatmap_run_tag=attn_heatmap_run_tag,
            model_dtype=model_dtype,
        )
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.llm.set_id(self.id, self.role)
        self.constraint = self.prompt_set.get_constraint(self.role)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Dict],
        temporal_info:Dict[str,Dict],
        **kwargs,
    )->Dict[str, Any]:
        """Prepare prompts in default mode."""
        system_prompt = self.constraint
        core_segments = [
            self.make_prompt_segment(
                kind="core",
                text=self.prompt_set.get_answer_prompt(
                    question=raw_inputs["task"],
                    role=self.role,
                ),
            )
        ]
        current_segments = []
        past_segments = []
        for id, info in spatial_info.items():
            current_segments.append(
                self.make_prompt_segment(
                    kind="current",
                    agent_id=id,
                    role=info.get("role"),
                    text=f"Agent {id} as a {info['role']} his answer to this question is:\n\n{info['output']}\n\n",
                )
            )
        for id, info in temporal_info.items():
            past_segments.append(
                self.make_prompt_segment(
                    kind="past",
                    agent_id=id,
                    role=info.get("role"),
                    text=f"Agent {id} as a {info['role']} his answer to this question was:\n\n{info['output']}\n\n",
                )
            )
        if current_segments:
            current_segments = [
                self.make_prompt_segment(
                    kind="core",
                    text="At the same time, there are the following responses to the same question for your reference:\n\n",
                ),
                *current_segments,
                self.make_prompt_segment(kind="core", text=" \n\n"),
            ]
        if past_segments:
            past_segments = [
                self.make_prompt_segment(
                    kind="core",
                    text="In the last round of dialogue, there were the following responses to the same question for your reference: \n\n",
                ),
                *past_segments,
            ]
        return self.build_prompt_payload(
            system_prompt=system_prompt,
            core_segments=core_segments,
            past_segments=past_segments,
            current_segments=current_segments,
        )

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
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
        message = self.build_llm_messages(inputs)
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        request_uid = input.get("_request_uid")
        inputs = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            **kwargs,
        )
        message = self.build_llm_messages(inputs)
        result = await self.llm.agen(
            message,
            request_uid=request_uid,
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
            round_index=input.get("_round_index"),
        )
        if self.role == "Programming Expert":
            answer = execute_code_get_return(result.text.lstrip("```python\n").rstrip("\n```"))
            self.record_tool_output(
                input,
                tool_name="execute_code_get_return",
                tool_output_text=answer,
            )
            result.text += f"\nthe answer is {answer}"
        return result
