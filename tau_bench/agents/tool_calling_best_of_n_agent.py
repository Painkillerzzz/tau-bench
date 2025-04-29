# Copyright Sierra

import json
import numpy as np
from litellm import completion
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


class ToolCallingBestOfNAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 1.0,
        n: int = 5,
        top_logprobs: int = 10,
        best_of_n_criterion: str = "perplexity",
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.n = n
        self.top_logprobs = top_logprobs
        self.best_of_n_criterion = best_of_n_criterion

    def _get_expand_num(self):
        """
        Get the number of responses to expand.
        """
        if self.n > 1:
            return self.n
        else:
            return 1

    def _compute_self_certainty(self, log_info, V):
        """
        log_info: List[List[List[float]]], shape = (resp_num, token_num, top_count)
        V: top logprobs to consider

        Return: ndarray of shape (resp_num,)
        """
        self_certainty = [
            -sum(sum(token) for token in choice) / (len(choice) * V)
            for choice in log_info
        ]

        return np.array(self_certainty)
    
    def _compute_perplexity(self, log_info):
        """
        Args:
            log_info: List[List[List[float]]], shape = (resp_num, token_num, top_count)

        Returns:
            perplexity: ndarray, shape = (resp_num,)
        """
        mean_logp = [
            -sum(token[0] for token in choice) / len(choice)
            for choice in log_info
        ]
        perplexity = np.exp(np.array(mean_logp))  # shape = (resp_num,)

        return perplexity
    
    def _select_best_response_by_metric(self, log_info, metric: str, V=5):
        """
        Args:
            resp_list: List[str] responses
            log_info: List[List[List[float]]] shape = (resp_num, token_num, top_count)
            metric: str, one of ['self_certainty', 'perplexity']
            V: top-V logprobs to consider (only used in self-certainty)

        Returns:
            (best_idx: int, score: float)
        """
        metric_fn_dict = {
            "perplexity": lambda: self._compute_perplexity(log_info),
            "self_certainty": lambda: self._compute_self_certainty(log_info, V),
        }

        if metric not in metric_fn_dict:
            raise ValueError(f"Unsupported metric: {metric}")

        scores = metric_fn_dict[metric]()
        best_idx = np.argmax(scores) if metric != "perplexity" else np.argmin(scores)
        return best_idx, scores[best_idx]

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        for _ in range(max_num_steps):
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
                n=self._get_expand_num(),
                logprobs=True,
                top_logprobs=self.top_logprobs,
            )
            # print(res.choices[0].logprobs.model_dump())
            log_info = []

            for i, choice in enumerate(res.choices):
                if choice is None:
                    raise ValueError(f"[ERROR] res.choices[{i}] is None")

                if not hasattr(choice, "logprobs") or choice.logprobs is None:
                    raise ValueError(f"[ERROR] res.choices[{i}].logprobs is None")

                if not hasattr(choice.logprobs, "content") or choice.logprobs.content is None:
                    print(choice.logprobs.model_dump())
                    print(choice.message.model_dump())
                    raise ValueError(f"[ERROR] res.choices[{i}].logprobs.content is None")

                choice_log = []
                for j, token in enumerate(choice.logprobs.content):
                    if token is None:
                        raise ValueError(f"[ERROR] res.choices[{i}].logprobs.content[{j}] is None")

                    if not hasattr(token, "top_logprobs"):
                        raise ValueError(f"[ERROR] token at res.choices[{i}].logprobs.content[{j}] missing top_logprobs attribute")

                    if token.top_logprobs is None:
                        raise ValueError(f"[ERROR] res.choices[{i}].logprobs.content[{j}].top_logprobs is None")

                    token_logprobs = []
                    for k, top_item in enumerate(token.top_logprobs):
                        if top_item is None:
                            raise ValueError(f"[ERROR] res.choices[{i}].logprobs.content[{j}].top_logprobs[{k}] is None")
                        if not hasattr(top_item, "logprob"):
                            raise ValueError(f"[ERROR] top_logprobs[{k}] missing 'logprob' at token[{j}] in choice[{i}]")

                        token_logprobs.append(top_item.logprob)

                    choice_log.append(token_logprobs)

                log_info.append(choice_log)

            best_idx, score = self._select_best_response_by_metric(
                log_info, metric=self.best_of_n_criterion, V=self.top_logprobs
            )

            next_message = res.choices[best_idx].message.model_dump()
            print(f"Best response index: {best_idx}, score: {score}")
            print(f"Best response: {next_message}")
            total_cost += res._hidden_params["response_cost"]
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
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
