import json
from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.modules.base import Module
from tau_bench.types import (
    Action,
    ProcessResult,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from typing import Optional, List, Dict, Any, Tuple


class Rethinker(Module):
    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.prompt = RETHINKER_INSTRUCTION
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def process(self, plan_step: Dict[str, Any], result: Dict[str, Any]) -> ProcessResult:
        """
        Given a plan step and the corresponding interaction result, decide next action.

        Args:
            plan_elem: One step from the plan list (dict).
            info: The info dictionary for the step result (must have "input" and "output").

        Returns:
            A string decision: REPLAN / CONTINUE.
        """

        user_prompt = f"""# Current Step
{json.dumps(plan_step, indent=2)}

# Execution Result
{json.dumps(result, indent=2)}
"""

# Task
# Analyze whether the execution result is correct and sufficient. 
# Respond with exactly one of: "REPLAN", "INTERACT", or "USETOOL" based on the following rules:

# - REPLAN: If the plan step is fundamentally wrong, or the goal has changed, or the tool call/user reply is irrelevant.
# - INTERACT: If more user information is needed to proceed (e.g., incomplete user reply, ambiguous answer).
# - USETOOL: If the result is good, correct, and the agent should proceed to next tool use or interaction.

# Only output the decision word in your response, nothing else.

        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user_prompt}
        ]

        res = completion(
            messages=messages,
            model=self.model,
            custom_llm_provider=self.provider,
            temperature=self.temperature,
        )

        message_text = res.choices[0].message.content.strip()
        message = res.choices[0].message.model_dump()

        try:
            rethink_step = json.loads(message_text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse rethinker output as JSON:\n{message_text}")

        if not isinstance(rethink_step, dict):
            raise ValueError(f"Planner output is not a dict:\n{rethink_step}")

        # Basic validation for step
        for key in ["reason", "decision"]:
            if key not in rethink_step:
                raise ValueError(f"Missing key '{key}' in rethink step: {rethink_step}")

        return ProcessResult(
            result=rethink_step,
            message=message,
            cost=res._hidden_params["response_cost"]
        )
    

RETHINKER_INSTRUCTION = """
# Instruction
You are the Rethinker Module. Your job is to analyze the last user interaction or tool execution result and decide the next action: continue or replan.

Your generation should have exactly the following format:

{
    "reason": <One-line explanation of why you made this decision. Be concise, no extra lines.>, 
    "decision": <One word: CONTINUE or REPLAN>
}

# Definitions:
- CONTINUE: If the tool execution was successful and goal is achieved.
- REPLAN: If the goal itself needs to be reconsidered (e.g., wrong approach, major error).

# Examples:
If the tool returned valid information matching the goal:
{
    "reason": "Tool returned the correct weather information.", 
    "decision": "CONTINUE"
}

If the user's intent cannot be fulfilled by any tool, or needs different planning:
{
    "reason": "The user's goal requires a different approach beyond available tools.", 
    "decision": "REPLAN"
}

Be strict and concise. Always output exactly the two fields in the specified format.
"""