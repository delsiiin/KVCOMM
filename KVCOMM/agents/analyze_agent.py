import os
from typing import Any,Dict
import re
import asyncio

import torch
from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.tools.search.wiki import search_wiki_main
from KVCOMM.llm.config import KVCommConfig
WIKI_TOKEN_LENGTH = int(os.environ.get('WIKI_TOKEN_LENGTH', 1024))

def find_strings_between_pluses(text):
    return re.findall(r'@+([^@]+?)@+', text)

@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(Node):
    """Research/analysis agent that can search wiki and compose context."""
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
    ):
        super().__init__(id, "AnalyzeAgent" ,domain, llm_name)
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
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.llm.set_id(self.id, self.role)
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.wiki_summary = ""

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Dict],
        temporal_info:Dict[str,Dict],
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare prompts with default execution mode."""
        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n"
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            if self.role == 'Wiki Searcher' and info['role']=='Knowledgeable Expert':
                queries = find_strings_between_pluses(info['output'])
                wiki = await search_wiki_main(queries)
                if len(wiki):
                    self.wiki_summary = ".\n".join(wiki)
                    token_ids = self.llm.tokenizer(self.wiki_summary, return_tensors="pt", add_special_tokens=False)
                    token_ids = {k: v[:, :WIKI_TOKEN_LENGTH].to(self.llm.model.device) for k, v in token_ids.items() if isinstance(v, torch.Tensor)}
                    self.wiki_summary = self.llm.tokenizer.decode(token_ids['input_ids'][0], skip_special_tokens=True)
                    user_prompt += f"The key entities of the problem are explained in Wikipedia as follows:{self.wiki_summary}"
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
        )
        if self.wiki_summary != "":
            self.wiki_summary = ""
        return result
