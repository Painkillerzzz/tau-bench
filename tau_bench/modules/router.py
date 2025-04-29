import json
from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.modules.base import Module
from tau_bench.types import (
    Action,
    ProcessResult
)
from typing import Optional, List, Dict, Any, Tuple


class Router(Module):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.prompt = ROUTER_INSTRUCTION
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.tools_info = tools_info

    def process(self, description: str, knowledge: List[Dict[str, Any]] = []) -> ProcessResult:
        """
        Process a single plan step (tool_use type) and generate an Action.

        Args:
            plan_step (dict): Contains description of what to do.

        Returns:
            Action object with tool name and arguments.
        """

        messages = [
            {"role": "system", "content": self.prompt},
        ]
        if len(knowledge) > 0:
            messages.append({
                "role": "system", 
                "content": f"# Knowledge: \n{json.dumps(knowledge)}"
            })

        messages.append({"role": "user", "content": f"# Goal description: {description}\n# Action: \n"})

        # print(f"Router messages: {messages}")

        res = completion(
            messages=messages,
            model=self.model,
            custom_llm_provider=self.provider,
            tools=self.tools_info,
            tool_choice="required",
            temperature=self.temperature,
        )

        message = res.choices[0].message.model_dump()

        try:
            action = message_to_action(message)
        except Exception as e:
            raise ValueError(f"Router failed to parse Action output.\nModel output: {message}\nError: {e}")

        return ProcessResult(
            result=action,
            message=message,
            cost=res._hidden_params["response_cost"]
        )


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    tool_call = message["tool_calls"][0]
    return Action(
        name=tool_call["function"]["name"],
        kwargs=json.loads(tool_call["function"]["arguments"]),
    )


ROUTER_INSTRUCTION = """
# Instruction
You are the Router Module. Your job is to select the most appropriate tool and generate a structured function call based on the given goal.

Your generation should have exactly the following format:

{"name": <The name of the selected tool>, "arguments": <The arguments to the tool in valid JSON format>}

- You must ONLY select from the available tools. You must NOT invent non-existing tools.
- If the description mentions a specific tool name, you MUST use exactly that tool. Do not choose another one, even if it seems related.
- You should not use made-up or placeholder arguments.
- The Action will be parsed, so it must be valid JSON.

For example, if the goal is "Find out the current weather in San Francisco" and there is a tool:
```json
{
    "function": {
        "name": "get_current_weather",
        "parameters": {
            "location": {"type": "string"},
            "format": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location", "format"]
    }
}
```
Your response should be: {"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}"""