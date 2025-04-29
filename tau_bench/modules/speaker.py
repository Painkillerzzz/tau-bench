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


class Speaker(Module):
    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.prompt = SPEAKER_INSTRUCTION
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def process(self, description: str, knowledge: List[Dict[str, Any]] = []) -> ProcessResult:
        """Given a planner's description of user interaction, generate the actual user-facing text."""
        system_prompt = self.prompt
        user_prompt = f"# Description: {description}\n# Output: "
        messages=[
            {"role": "system", "content": system_prompt},
        ]
        
        if len(knowledge) > 0:
            messages.append({
                "role": "system", 
                "content": f"# Knowledge: \n{json.dumps(knowledge)}"
            })
        messages.append({"role": "user", "content": user_prompt})

        res = completion(
            messages=messages,
            model=self.model,
            custom_llm_provider=self.provider,
            temperature=self.temperature,
        )

        message_text = res.choices[0].message.content.strip()
        message = res.choices[0].message.model_dump()

        return ProcessResult(
            result=Action(
                name=RESPOND_ACTION_NAME,
                kwargs={RESPOND_ACTION_FIELD_NAME: message_text},
            ),
            message=message,
            cost=res._hidden_params["response_cost"]
        )

SPEAKER_INSTRUCTION = """
# Instruction
You are the Speaker Module. Your task is to generate a user-facing natural language message based on a provided description.

- If the description suggests asking the user for missing information, you should frame it as a clear, polite question.
- If the description suggests informing the user, you should provide a concise and friendly statement.
- Keep your generation as short and clear as possible.
- Do NOT invent new information beyond the given description.
- Avoid meta-comments like "According to your request..." or "As per the description..."

Example 1:
Description: Ask the user whether they prefer celsius or fahrenheit.
Output: Would you prefer the temperature in Celsius or Fahrenheit?

Example 2:
Knowledge: The product price is $499.
Description: Inform the user about the product price.
Output: The price of the product is $499.

Always directly and politely address the user."""
