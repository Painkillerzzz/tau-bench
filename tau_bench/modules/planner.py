import json
from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.modules.base import Module
from tau_bench.types import (
    ProcessResult,
)
from typing import Optional, List, Dict, Any, Tuple


class Planner(Module):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.prompt = (
            wiki + "\n# Available tools\n" + json.dumps(tools_info) + PLANNER_INSTRUCTION
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.tools_info = tools_info

    def process(self, request: str, knowledge: List[Dict[str, Any]] = [], failed_step: Optional[Dict[str, Any]] = None) -> ProcessResult:
        """
        Given conversation history (messages), generate a multi-step plan.
        
        Args:
            messages: List of conversation messages (system/user/tool format)

        Returns:
            List of step dictionaries, each with keys: step, thought, action_type, description
        """
        messages = [{"role": "system", "content": self.prompt}]

        if len(knowledge) > 0:
            messages.append({
                "role": "system", 
                "content": f"# Knowledge: \n{json.dumps(knowledge)}"
            })

        if failed_step:
            messages.append({
                "role": "system",
                "content": f"# Failed step: \n{json.dumps(failed_step)}"
            })

        messages.append({"role": "user", "content": f"# Request: \n{request}"})

        # print(f"Knowledge: {json.dumps(knowledge)}")

        messages.append({"role": "user", "content": f"# Request: \n{request}"})

        # print(f"Planner messages: {messages}")

        res = completion(
            messages=messages,
            model=self.model,
            custom_llm_provider=self.provider,
            temperature=self.temperature,
        )
        # print(f"Planner response: {res.choices[0].message}")

        message_text = res.choices[0].message.content.strip()
        import re
        message_text = re.sub(r"^```json\s*\n", "", message_text) 
        message_text = re.sub(r"\n```$", "", message_text)
        message = res.choices[0].message.model_dump()

        try:
            plan_steps = json.loads(message_text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse planner output as JSON:\n{message_text}")

        if not isinstance(plan_steps, list):
            raise ValueError(f"Planner output is not a list:\n{plan_steps}")

        # Basic validation for each step
        for idx, step in enumerate(plan_steps):
            if not isinstance(step, dict):
                raise ValueError(f"Step {idx} is not a dict: {step}")
            for key in ["thought", "action_type", "description"]:
                if key not in step:
                    raise ValueError(f"Missing key '{key}' in step {idx}: {step}")

        return ProcessResult(
            result=plan_steps,
            message=message,
            cost=res._hidden_params["response_cost"],
        )



PLANNER_INSTRUCTION = """
# Instruction
You are the Planning Module. Your task is to create a multi-step plan to solve the user's request or complete the current task, using available tools when necessary.

For each step, output the following fields:
- "thought": A one-line reasoning why this step is needed.
- "action_type": Either "tool_use" (using a tool) or "user_interaction" (directly interacting with user).
- "description": A short description of what action to perform.

Notes:
- You should provide a multi-step plan, not just a single step. Your plan should be a list of steps.
- Check the "Knowledge" field for any relevant information that can help in planning. Don't repeat it in the plan.
- Check the "Available tools" field for the list of tools you can use. Use them when necessary.
- Check the "Failed step" field for any previous steps that did not succeed. Use this information to adjust your plan.
- If information is missing, you should still assume reasonable defaults to generate a complete multi-step plan, remember to plan a "user_interaction" step to ask the user.
- Tool use must reference the tool name and what it is used for.
- User interaction must specify what to ask or what to inform.

Format your entire output as a valid JSON array of step objects.

For example, for the query "Get the weather in San Francisco" without a specified format, you should plan:

```json
[
    {
        "thought": "The user did not specify temperature format, I need to ask them.",
        "action_type": "user_interaction",
        "description": "Ask the user whether they prefer celsius or fahrenheit."
    },
    {
        "thought": "After knowing the format, I can fetch the weather information.",
        "action_type": "tool_use",
        "description": "Use `get_current_weather` to get the weather of San Francisco with the specified format."
    },
    {
        "thought": "Finally, summarize the result to the user.",
        "action_type": "user_interaction",
        "description": "Respond to the user with the weather information."
    }
]
```
Be concise, precise, and generate valid JSON. """