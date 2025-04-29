# Copyright Sierra

import json
from litellm import completion
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.modules.base import Module
from tau_bench.modules.planner import Planner
from tau_bench.modules.router import Router
from tau_bench.modules.rethinker import Rethinker
from tau_bench.modules.speaker import Speaker
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME, RESPOND_ACTION_FIELD_NAME

MODULE_MAP: Dict[str, Module] = {
    "planner": Planner,
    "router": Router,
    "rethinker": Rethinker,
    "speaker": Speaker,
}

class ToolCallingModularAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        modules_info: Dict[str, Dict[str, Any]],
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.modules = {}
        for module_name, module_info in modules_info.items():
            assert module_name in MODULE_MAP.keys(), f"module {module_name} not in {MODULE_MAP.keys()}"
            
            module_class = MODULE_MAP[module_name]
            init_params = module_info.copy()
            init_params["tools_info"] = self.tools_info
            init_params["wiki"] = self.wiki
            
            module_init_args = module_class.__init__.__code__.co_varnames
            filtered_params = {k: v for k, v in init_params.items() if k in module_init_args}
            
            self.modules[module_name] = module_class(**filtered_params)

        self.plan: List[Dict[str, Any]] = []
        self.knowledge: List[Dict[str, Any]] = []

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation

        print(f"Initial observation: {obs}")
        # raise NotImplementedError("ToolCallingModularAgent does not support env.reset()")
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        planner: Planner = self.modules["planner"]

        planner_result = planner.process(obs, self.knowledge)
        self.plan = planner_result.result
        messages.append(planner_result.message)
        total_cost += planner_result.cost

        print(f"Initial plan: \n{json.dumps(self.plan, indent=2)}")

        plan_step_idx = 0

        for _ in range(max_num_steps):
            if plan_step_idx >= len(self.plan):
                planner_result = planner.process(obs, self.knowledge)
                self.plan += planner_result.result
                total_cost += planner_result.cost
                
                print(f"Extended plan: \n{json.dumps(self.plan, indent=2)}")

            current_step = self.plan[plan_step_idx]
            action_type = current_step["action_type"]

            print(f"\n---\nCurrent step {plan_step_idx}: {json.dumps(current_step, indent=2)}")

            if action_type == "user_interaction":
                print("\nUser interaction step")
                speaker: Speaker = self.modules["speaker"]
                speaker_result = speaker.process(current_step["description"])

                action = speaker_result.result
                message = speaker_result.message
                total_cost += speaker_result.cost

                print(f"\nAction: {action}")

                env_response = env.step(action)

                print(f"\nEnv response: {env_response.observation}")

                messages.extend(
                    [
                        message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
                new_knowledge = {
                    "type": "user_interaction",
                    "input": action.kwargs[RESPOND_ACTION_FIELD_NAME],
                    "output": env_response.observation,
                }
                self.knowledge.append(new_knowledge)

                rethinker: Rethinker = self.modules["rethinker"]
                rethinker_result = rethinker.process(plan_step=current_step, result=new_knowledge)
                decision = rethinker_result.result
                total_cost += rethinker_result.cost

                print(f"\nRethinker decision: \n{json.dumps(decision, indent=2)}")

                if decision["decision"] == "REPLAN":
                    planner_result = planner.process(obs, self.knowledge, decision) # TODO: replan input
                    self.plan = self.plan[:plan_step_idx] + planner_result.result
                    total_cost += planner_result.cost
                    print(f"Replanned plan: \n{json.dumps(self.plan, indent=2)}")
                else:
                    plan_step_idx += 1
                # else:
                #     raise ValueError(f"Unexpected rethinker decision: {decision}")

            elif action_type == "tool_use":
                print("\nTool use step")
                router: Router = self.modules["router"]
                router_result = router.process(current_step["description"], self.knowledge)
                
                action = router_result.result
                message = router_result.message
                total_cost += router_result.cost

                print(f"\nAction: {action}")

                env_response = env.step(action)

                print(f"\nEnv response: {json.dumps(env_response.observation, indent=2)}")

                messages.extend(
                    [
                        message,
                        {
                            "role": "tool",
                            "tool_call_id": message["tool_calls"][0]["id"],
                            "name": action.name,
                            "content": env_response.observation,
                        },
                    ]
                )

                new_knowledge = {
                    "type": "tool_use",
                    "input": action.model_dump(),
                    "output": env_response.observation,
                }

                if "Error" in env_response.observation:
                    print("\n---\nTool error step")
                    rethinker: Rethinker = self.modules["rethinker"]
                    rethinker_result = rethinker.process(plan_step=current_step, result=new_knowledge)

                    decision = rethinker_result.result
                    messages.append(rethinker_result.message)
                    total_cost += rethinker_result.cost

                    print(f"\nRethinker decision: {json.dumps(decision, indent=2)}")

                    if decision["decision"] == "REPLAN":
                        planner_result = planner.process(obs, self.knowledge, decision)
                        self.plan = self.plan[:plan_step_idx] + planner_result.result
                        total_cost += planner_result.cost
                        print(f"Replanned plan: \n{json.dumps(self.plan, indent=2)}")
                    else:
                        plan_step_idx += 1
                else:
                    # Tool success, move on without rethink
                    self.knowledge.append(new_knowledge)
                    plan_step_idx += 1
            else:
                raise ValueError(f"Unknown action_type {action_type} in plan step.")

            # Common logging and state updates
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
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
