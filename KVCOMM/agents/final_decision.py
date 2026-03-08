import asyncio
from typing import Any, Dict, List
from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.tools.coding.python_executor import execute_code_get_return, PyExecutor
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.log import logger

@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode(Node):
    """Final code synthesis agent that integrates peers and executes tests."""
    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
        compress_mode: bool = False,
        compress_method: str = "rkv",
        compress_budget: int = 1024,
        compress_divide_length: int = 128,
    ):
        super().__init__(id, "FinalWriteCode" ,domain, llm_name)
        prefix = ""
        self.llm = LLMRegistry.get(
            llm_name,
            prefix=prefix,
            llm_config=llm_config,
            compress_mode=compress_mode,
            compress_method=compress_method,
            compress_budget=compress_budget,
            compress_divide_length=compress_divide_length,
        )
        self.role = 'FinalWriteCode'
        self.llm.set_id(self.id, 'FinalWriteCode')
        self.prompt_set = PromptSetRegistry.get(domain)
        self._executor = PyExecutor()

    @staticmethod
    def extract_example(prompt: Dict[str, Any] | str) -> List[str]:
        """Return doctest-style assertions extracted from the task description."""
        if isinstance(prompt, dict):
            prompt_text = str(prompt.get("task", ""))
        else:
            prompt_text = str(prompt)

        lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
        results: List[str] = []
        iterator = iter(lines)
        for line in iterator:
            if not line.startswith(">>>"):
                continue
            function_call = line[4:].strip()
            expected_output = next(iterator, "").strip()
            if not function_call or not expected_output:
                continue
            results.append(f"assert {function_call} == {expected_output}")
        return results

    @staticmethod
    def _is_python_code_block(text: str) -> bool:
        text = text.strip()
        return text.startswith("```python") and text.endswith("```")

    @staticmethod
    def _extract_python_code(text: str) -> str:
        """Extract pure python code from a fenced block."""
        if not FinalWriteCode._is_python_code_block(text):
            return text
        content = text.strip()[len("```python") :].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
        return content

    def _summarize_agent_outputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Any],
        internal_tests: List[str],
    ) -> str:
        """Summarize peer outputs, running tests on code blocks when present."""
        paragraphs: List[str] = []
        for agent_id, info in spatial_info.items():
            role = info.get("role", "agent")
            output = info.get("output")
            if not isinstance(output, str):
                paragraphs.append(f"Agent {agent_id} as a {role} returned invalid output.\n\n")
                continue
            if self._is_python_code_block(output):
                code = self._extract_python_code(output)
                is_solved, feedback, _ = self._executor.execute(code, internal_tests, timeout=10)
                paragraphs.append(
                    f"Agent {agent_id} as a {role}:\n\n"
                    f"The code written by the agent is:\n\n{output}\n\n"
                    f"Whether it passes internal testing? {is_solved}.\n\n"
                    f"The feedback is:\n\n{feedback}.\n\n"
                )
                continue
            paragraphs.append(
                f"Agent {agent_id} as a {role} provides the following info: {output}\n\n"
            )
        return "".join(paragraphs)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Any],
        temporal_info:Dict[str,Any],
        **kwargs,
    )->Dict[str, Any]:
        """Process the raw inputs in default mode."""
        system_prompt = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{system_prompt}.\n {self.constraint}"
        prefix_text = kwargs.get("prefix", "")
        spatial_str = ""
        if self.domain in {"gsm8k", "mmlu"}:
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if info["role"] == "Programming Expert":
                    answer = execute_code_get_return(
                        info["output"].split("```python\n")[-1].split("\n```")[0]
                    )
                    agent_output += f"\n the result is {answer}"
                spatial_str += (
                    f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                )
        elif self.domain == "humaneval":
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if (
                    self.role not in {"Normal Programmer", "Stupid Programmer"}
                    and info["role"] != "Algorithm Designer"
                ):
                    code = info["output"].split("```python\n")[-1].split("\n```")[0]
                    is_solved, feedback, _ = PyExecutor().execute(
                        code, getattr(self, "internal_tests", []), timeout=10
                    )
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                        f"{agent_output}\n\n Whether it passes internal testing?\n{is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
                    )
                else:
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                    )

        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = (
            f"{decision_few_shot} {prefix_text} {raw_inputs['task']}\n At the same time, the output of other agents is as follows:\n\n"
            f"{spatial_str}\n\n"
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

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
        message = [
            {"role": "system", "content": inputs["system_prompt"]},
            {"role": "user", "content": inputs["user_prompt"]},
        ]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        request_uid = input.get("_request_uid")
        inputs = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            **kwargs,
        )
        message = [
            {"role": "system", "content": inputs["system_prompt"]},
            {"role": "user", "content": inputs["user_prompt"]},
        ]
        result = await self.llm.agen(
            message,
            request_uid=request_uid,
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
        )
        return result


@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    """Final referencing/answer agent assembling the final response."""
    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
        compress_mode: bool = False,
        compress_method: str = "rkv",
        compress_budget: int = 1024,
        compress_divide_length: int = 128,
    ):
        super().__init__(id, "FinalRefer" ,domain, llm_name)
        prefix = ""
        self.llm = LLMRegistry.get(
            llm_name,
            prefix=prefix,
            llm_config=llm_config,
            compress_mode=compress_mode,
            compress_method=compress_method,
            compress_budget=compress_budget,
            compress_divide_length=compress_divide_length,
        )
        self.role = 'FinalRefer'
        self.llm.set_id(self.id, 'FinalRefer')
        self.prompt_set = PromptSetRegistry.get(domain)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Any],
        temporal_info:Dict[str,Any],
        **kwargs,
    )->Dict[str, Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        system_prompt = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{system_prompt}.\n {self.constraint}"
        prefix_text = kwargs.get("prefix", "")
        spatial_str = ""
        if self.domain in {"gsm8k", "mmlu"}:
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if info["role"] == "Programming Expert":
                    answer = execute_code_get_return(
                        info["output"].split("```python\n")[-1].split("\n```")[0]
                    )
                    agent_output += f"\n the result is {answer}"
                spatial_str += (
                    f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                )
        elif self.domain == "humaneval":
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if (
                    self.role not in {"Normal Programmer", "Stupid Programmer"}
                    and info["role"] != "Algorithm Designer"
                ):
                    code = info["output"].split("```python\n")[-1].split("\n```")[0]
                    is_solved, feedback, _ = PyExecutor().execute(
                        code, getattr(self, "internal_tests", []), timeout=10
                    )
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                        f"{agent_output}\n\n Whether it passes internal testing?\n{is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
                    )
                else:
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                    )

        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = (
            f"{decision_few_shot} {prefix_text} {raw_inputs['task']}\n At the same time, the output of other agents is as follows:\n\n"
            f"{spatial_str}\n\n"
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

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

        inputs = asyncio.run(
            self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                **kwargs,
            )
        )
        message = [{'role':'system','content':inputs["system_prompt"]},{'role':'user','content':inputs["user_prompt"]}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """Handle asynchronous execution across different KV cache strategies."""
        if self.domain == 'humaneval':
            self.internal_tests = self.extract_example(input)
        request_uid = input.get("_request_uid")
        inputs = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            **kwargs,
        )
        message = [{'role':'system','content':inputs["system_prompt"]},{'role':'user','content':inputs["user_prompt"]}]
        result = await self.llm.agen(
            message,
            request_uid=request_uid,
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
        )
        return result

@AgentRegistry.register('FinalDirect')
class FinalDirect(Node):
    def __init__(
        self,
        id: str | None =None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
        compress_mode: bool = False,
        compress_method: str = "rkv",
        compress_budget: int = 1024,
        compress_divide_length: int = 128,
    ):
        """ Used for Directed IO """
        super().__init__(id, "FinalDirect")
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output


@AgentRegistry.register('FinalMajorVote')
class FinalMajorVote(Node):
    def __init__(
        self,
        id: str | None =None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
        compress_mode: bool = False,
        compress_method: str = "rkv",
        compress_budget: int = 1024,
        compress_divide_length: int = 128,
    ):
        """ Used for Directed IO """
        super().__init__(id, "FinalMajorVote")
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = self.prompt_set.postprocess_answer(info['output'])
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = await self.prompt_set.postprocess_answer(info['output'])
            logger.debug("Processed output: {}", processed_output)
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output
