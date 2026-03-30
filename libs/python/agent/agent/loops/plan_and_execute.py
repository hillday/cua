"""
Plan-and-Execute agent loop implementation.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import litellm

from ..agent import robust_json_loads
from ..decorators import find_agent_config, register_agent
from ..loops.base import AsyncAgentConfig
from ..responses import convert_responses_items_to_completion_messages
from ..types import AgentCapability

logger = logging.getLogger(__name__)


@register_agent(r"^plan-and-execute/.*", priority=10)
class PlanAndExecuteConfig(AsyncAgentConfig):
    """
    Plan-and-Execute agent configuration.
    Uses three agents: Planner, Executor, Supervisor.
    """

    def __init__(self):
        self.plan: List[str] = []
        self.current_step_index: int = 0
        self.last_feedback: str = ""
        self.executor_loop: Optional[AsyncAgentConfig] = None

    async def _call_planner(
        self, messages: List[Dict[str, Any]], model: str, **kwargs
    ) -> List[str]:
        completion_messages = convert_responses_items_to_completion_messages(
            messages, allow_images_in_tool_results=True
        )

        system_prompt = """You are a Planner Agent for a Computer Use task.
Your job is to break down the user's request into a list of specific, simple steps.
The Executor agent will carry out these steps one by one. Each step should be a simple UI operation (like 'click the search bar', 'type query', 'press enter').
Also, consider any skills or knowledge base you might have (like screenshots context).
Output a JSON object with a 'plan' array containing strings.
Example: {"plan": ["Click the start menu", "Type 'calculator'", "Click the Calculator app", "Wait for it to open", "Click '5'", "Click '+'", "Click '7'", "Click '='"]}
"""

        planner_messages = [{"role": "system", "content": system_prompt}]
        planner_messages.extend(completion_messages)

        # Filter kwargs to only pass what litellm.acompletion accepts
        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["api_key", "api_base", "max_retries", "temperature", "top_p"]
        }

        try:
            response = await litellm.acompletion(
                model=model,
                messages=planner_messages,
                response_format={"type": "json_object"},
                **valid_kwargs,
            )

            content = response.choices[0].message.content
            plan_data = robust_json_loads(content)
            return plan_data.get("plan", ["Execute user request"])
        except Exception as e:
            logger.error(f"Planner failed: {e}")
            return ["Execute user request"]

    async def _call_supervisor(
        self, messages: List[Dict[str, Any]], current_step: str, model: str, **kwargs
    ) -> Dict[str, Any]:
        # Get the recent history (e.g. last 5 messages to save tokens but keep the last action and its result)
        recent_messages = messages[-5:] if len(messages) >= 5 else messages
        completion_messages = convert_responses_items_to_completion_messages(
            recent_messages, allow_images_in_tool_results=True
        )

        system_prompt = f"""You are a Supervisor Agent.
Your job is to evaluate the result of the Executor's last action based on the screenshot and action history.
Current Step to accomplish: {current_step}

Did the Executor successfully complete the action? Is the current step fully completed?
Output a JSON object with:
- 'status': one of 'action_success', 'action_failure', 'step_complete', 'task_complete'
- 'feedback': string with brief feedback for the Planner/Executor
"""

        supervisor_messages = [{"role": "system", "content": system_prompt}]
        supervisor_messages.extend(completion_messages)

        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["api_key", "api_base", "max_retries", "temperature", "top_p"]
        }

        try:
            response = await litellm.acompletion(
                model=model,
                messages=supervisor_messages,
                response_format={"type": "json_object"},
                **valid_kwargs,
            )

            content = response.choices[0].message.content
            return robust_json_loads(content)
        except Exception as e:
            logger.error(f"Supervisor failed: {e}")
            return {"status": "action_success", "feedback": "Could not parse supervisor output."}

    async def predict_step(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: Optional[int] = None,
        stream: bool = False,
        computer_handler=None,
        use_prompt_caching: Optional[bool] = False,
        _on_api_start=None,
        _on_api_end=None,
        _on_usage=None,
        _on_screenshot=None,
        **kwargs,
    ) -> Dict[str, Any]:
        prefix = "plan-and-execute/"
        underlying_model = model[len(prefix) :] if model.startswith(prefix) else model

        if not self.executor_loop:
            config_info = find_agent_config(underlying_model)
            if not config_info:
                # Fallback to generic_vlm or openai if underlying model not found
                config_info = find_agent_config("openai/gpt-4o")
            self.executor_loop = config_info.agent_class()

        last_msg = messages[-1] if messages else {}

        is_new_task = False
        if last_msg.get("role") == "user":
            content = last_msg.get("content")
            if isinstance(content, str):
                is_new_task = True
            elif isinstance(content, list):
                has_text = any(item.get("type") == "text" for item in content)
                if has_text:
                    is_new_task = True

        # 1. Planner Phase
        if is_new_task or not self.plan:
            self.plan = await self._call_planner(messages, underlying_model, **kwargs)
            self.current_step_index = 0
            self.last_feedback = ""
            logger.info(f"Plan generated: {self.plan}")

        # 2. Supervisor Phase
        elif last_msg.get("type") == "computer_call_output" or (
            last_msg.get("role") == "user" and "input_image" in str(last_msg)
        ):
            if self.current_step_index < len(self.plan):
                current_step = self.plan[self.current_step_index]
                eval_result = await self._call_supervisor(
                    messages, current_step, underlying_model, **kwargs
                )

                self.last_feedback = eval_result.get("feedback", "")
                status = eval_result.get("status")
                logger.info(f"Supervisor evaluation: {status} - {self.last_feedback}")

                if status == "step_complete":
                    self.current_step_index += 1
                    self.last_feedback = ""
                    if self.current_step_index >= len(self.plan):
                        return {
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Task completed successfully according to the supervisor.",
                                        }
                                    ],
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                        }
                elif status == "task_complete":
                    return {
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Task completed successfully according to the supervisor.",
                                    }
                                ],
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    }
                elif status == "action_failure":
                    # For failure, we can let the Executor try again with feedback, or replan.
                    # Currently we just let Executor try again.
                    pass

        # 3. Executor Phase
        if self.current_step_index >= len(self.plan):
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "All planned steps completed."}],
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        executor_messages = messages.copy()
        current_step_desc = self.plan[self.current_step_index]

        instruction = f"You are the Executor Agent.\n"
        instruction += f"Overall Plan: {json.dumps(self.plan)}\n"
        instruction += (
            f"Current Step ({self.current_step_index + 1}/{len(self.plan)}): {current_step_desc}\n"
        )
        if self.last_feedback:
            instruction += f"Supervisor Feedback on last action: {self.last_feedback}\n"
        instruction += "Please take the next simple action using the provided tools to accomplish the current step."

        executor_messages.append(
            {"role": "user", "content": [{"type": "text", "text": instruction}]}
        )

        # Call the underlying executor loop
        return await self.executor_loop.predict_step(
            messages=executor_messages,
            model=underlying_model,
            tools=tools,
            max_retries=max_retries,
            stream=stream,
            computer_handler=computer_handler,
            use_prompt_caching=use_prompt_caching,
            _on_api_start=_on_api_start,
            _on_api_end=_on_api_end,
            _on_usage=_on_usage,
            _on_screenshot=_on_screenshot,
            **kwargs,
        )

    async def predict_click(
        self, model: str, image_b64: str, instruction: str, **kwargs
    ) -> Optional[Tuple[int, int]]:
        prefix = "plan-and-execute/"
        underlying_model = model[len(prefix) :] if model.startswith(prefix) else model

        if not self.executor_loop:
            config_info = find_agent_config(underlying_model)
            if config_info:
                self.executor_loop = config_info.agent_class()

        if self.executor_loop and hasattr(self.executor_loop, "predict_click"):
            return await self.executor_loop.predict_click(
                underlying_model, image_b64, instruction, **kwargs
            )
        return None

    def get_capabilities(self) -> List[AgentCapability]:
        return ["click", "step"]
