import asyncio
from typing import Any,Dict
from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.tools.coding.python_executor import PyExecutor
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.metrics import GenerationResult

@AgentRegistry.register('CodeWriting')
class CodeWriting(Node):
    """Programming agent that validates peers with internal tests."""
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
        super().__init__(id, "CodeWriting" ,domain, llm_name)
        prefix = ""
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
        self.constraint = self.prompt_set.get_constraint(self.role) 
        self.llm.set_id(self.id, self.role)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Dict],
        temporal_info:Dict[str,Dict],
        **kwargs,
    )->Dict[str, Any]:
        """Prepare prompts in default mode."""
        system_prompt = self.constraint
        current_segments = []
        past_segments = []
        for id, info in spatial_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```") and self.role != 'Normal Programmer' and self.role != 'Stupid Programmer':
                output = info['output'].split("```python\n")[-1].split("\n```")[0]
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                self.record_tool_output(
                    raw_inputs,
                    tool_name="PyExecutor.execute",
                    tool_output_text=feedback,
                    metadata={
                        "is_solved": bool(is_solved),
                        "source": "spatial",
                        "peer_agent_id": id,
                        "peer_agent_role": info.get("role"),
                    },
                )
                current_segments.append(
                    self.make_prompt_segment(
                        kind="current",
                        agent_id=id,
                        role=info.get("role"),
                        text=(
                            f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n "
                            f"Whether it passes internal testing?\n{is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
                        ),
                    )
                )
            else:
                current_segments.append(
                    self.make_prompt_segment(
                        kind="current",
                        agent_id=id,
                        role=info.get("role"),
                        text=f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n",
                    )
                )
        for id, info in temporal_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```") and self.role != 'Normal Programmer' and self.role != 'Stupid Programmer':
                output = info['output'].split("```python\n")[-1].split("\n```")[0]
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                self.record_tool_output(
                    raw_inputs,
                    tool_name="PyExecutor.execute",
                    tool_output_text=feedback,
                    metadata={
                        "is_solved": bool(is_solved),
                        "source": "temporal",
                        "peer_agent_id": id,
                        "peer_agent_role": info.get("role"),
                    },
                )
                past_segments.append(
                    self.make_prompt_segment(
                        kind="past",
                        agent_id=id,
                        role=info.get("role"),
                        text=(
                            f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n "
                            f"Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
                        ),
                    )
                )
            else:
                past_segments.append(
                    self.make_prompt_segment(
                        kind="past",
                        agent_id=id,
                        role=info.get("role"),
                        text=f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n",
                    )
                )
        core_segments = [
            self.make_prompt_segment(
                kind="core",
                text=f"The task is:\n\n{raw_inputs['task']}\n",
            )
        ]
        if current_segments:
            current_segments = [
                self.make_prompt_segment(
                    kind="core",
                    text="At the same time, the outputs and feedbacks of other agents are as follows:\n\n",
                ),
                *current_segments,
                self.make_prompt_segment(kind="core", text=" \n\n"),
            ]
        if past_segments:
            past_segments = [
                self.make_prompt_segment(
                    kind="core",
                    text="In the last round of dialogue, the outputs and feedbacks of some agents were: \n\n",
                ),
                *past_segments,
            ]
        return self.build_prompt_payload(
            system_prompt=system_prompt,
            core_segments=core_segments,
            past_segments=past_segments,
            current_segments=current_segments,
        )

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        self.internal_tests = self.extract_example(input)
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
        self.internal_tests = self.extract_example(input)
        request_uid = input.get("_request_uid")
        inputs = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            **kwargs,
        )
        system_prompt = inputs["system_prompt"]
        user_prompt = inputs["user_prompt"]
        if system_prompt == "is_solved":
            return GenerationResult(
                text=user_prompt,
                mode="default",
                ttft=0.0,
                metadata={
                    "input_char_len": len(system_prompt) + len(user_prompt),
                    "output_char_len": len(user_prompt),
                },
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
        return result
