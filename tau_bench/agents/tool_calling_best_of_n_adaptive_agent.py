# Copyright Sierra

import json
import numpy as np
from litellm import completion
from typing import List, Optional, Dict, Any

from tau_bench.agents.tool_calling_best_of_n_agent import ToolCallingBestOfNAgent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


class ToolCallingBestOfNAdaptiveAgent(ToolCallingBestOfNAgent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 1.0,
        n: int = 5,
        top_logprobs: int = 20,
        best_of_n_criterion: str = "perplexity",
    ):
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            temperature=temperature,
            n=n,
            top_logprobs=top_logprobs,
            best_of_n_criterion=best_of_n_criterion,
        )

    def _get_expand_num(self):
        raise NotImplementedError(
            "Adaptive agent does not support _get_expand_num. Please use the default n=1."
        )

def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
