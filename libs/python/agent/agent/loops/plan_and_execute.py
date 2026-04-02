"""
Plan-and-Execute agent loop implementation.
"""

import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

from ..agent import robust_json_loads
from ..decorators import find_agent_config, register_agent
from ..loops.base import AsyncAgentConfig
from ..responses import convert_responses_items_to_completion_messages
from ..tools.skill import SkillTool
from ..types import AgentCapability

logger = logging.getLogger(__name__)


def _extract_user_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("input_text", "text") and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _find_vision_kb_root() -> Optional[Path]:
    env_dir = os.getenv("CUA_VISION_KB_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists() and p.is_dir():
            return p

    candidates = [
        Path(os.getcwd()) / "vision_kb",
        Path(os.getcwd()).parent / "vision_kb",
        Path(__file__).resolve().parents[4] / "vision_kb",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def _find_skills_json() -> Optional[Path]:
    potential_paths = [
        Path(os.getcwd()) / "skills.json",
        Path(os.getcwd()).parent / "skills.json",
        Path(__file__).resolve().parents[4] / "skills.json",
    ]
    current_dir = Path(__file__).resolve().parent
    for _ in range(5):
        potential_paths.append(current_dir / "skills.json")
        current_dir = current_dir.parent
    for p in potential_paths:
        if p.exists() and p.is_file():
            return p
    return None


def _load_skills() -> Dict[str, dict]:
    path = _find_skills_json()
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_skills_base_dir() -> Path:
    skills_path = _find_skills_json()
    if skills_path and skills_path.exists():
        return skills_path.parent
    return Path(os.getcwd())


def _resolve_skill_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (_get_skills_base_dir() / p).resolve()


def _count_kb_images(kb_message: Optional[Dict[str, Any]]) -> int:
    if not kb_message:
        return 0
    content = kb_message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1 for item in content if isinstance(item, dict) and item.get("type") == "input_image"
    )


def _make_data_url(path: Path) -> Optional[str]:
    try:
        raw = path.read_bytes()
        if len(raw) > 2_000_000:
            return None
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/png"
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _build_skill_kb_message(
    skill_name: str, skill_info: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    kb_images = skill_info.get("kb_images")
    if isinstance(kb_images, list) and kb_images:
        max_images = int(
            skill_info.get("kb_max_images") or os.getenv("CUA_VISION_KB_MAX_IMAGES", "6")
        )
        content: List[Dict[str, Any]] = [
            {
                "type": "input_text",
                "text": f"[SkillKB:{skill_name}] 以下截图是该技能的操作参考资料，用于辅助规划、执行与监督；如果与当前页面不一致，以实时截图为准。",
            }
        ]
        loaded = 0
        for v in kb_images:
            if loaded >= max_images:
                break
            if not isinstance(v, str) or not v.strip():
                continue
            p = _resolve_skill_path(v)
            if not p.exists() or not p.is_file():
                continue
            url = _make_data_url(p)
            if not url:
                continue
            content.append({"type": "input_image", "image_url": url})
            loaded += 1
        if loaded == 0:
            logger.info(f"[SkillKB] kb_images 配置存在但无可用图片 skill={skill_name}")
            return None
        logger.info(f"[SkillKB] 已加载(kb_images) skill={skill_name} images={loaded}")
        return {"role": "user", "content": content}

    kb_dir_cfg = skill_info.get("kb_dir")
    if isinstance(kb_dir_cfg, str) and kb_dir_cfg.strip():
        kb_dir = _resolve_skill_path(kb_dir_cfg)
        if not kb_dir.exists() or not kb_dir.is_dir():
            logger.info(f"[SkillKB] kb_dir 不存在/不是目录 skill={skill_name} kb_dir={kb_dir}")
            return None
    else:
        root = _find_vision_kb_root()
        if not root:
            logger.info(f"[SkillKB] 未找到 vision_kb 根目录，跳过加载 skill={skill_name}")
            return None
        candidates: List[str] = [skill_name]
        aliases = skill_info.get("aliases", [])
        if isinstance(aliases, list):
            candidates.extend(str(a) for a in aliases if a)
        normalized_candidates: List[str] = []
        for c in candidates:
            if not isinstance(c, str):
                continue
            if c and c not in normalized_candidates:
                normalized_candidates.append(c)
            lowered = c.lower()
            if lowered and lowered not in normalized_candidates:
                normalized_candidates.append(lowered)

        kb_dir = None
        for name in normalized_candidates:
            p = root / name
            if p.exists() and p.is_dir():
                kb_dir = p
                break
        if not kb_dir:
            logger.info(
                f"[SkillKB] 未找到匹配目录，跳过加载 skill={skill_name} candidates={normalized_candidates}"
            )
            return None

    max_images = int(skill_info.get("kb_max_images") or os.getenv("CUA_VISION_KB_MAX_IMAGES", "6"))
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths = [p for p in sorted(kb_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        logger.info(f"[SkillKB] 目录无可用图片 skill={skill_name} dir={kb_dir}")
        return None

    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": f"[SkillKB:{skill_name}] 以下截图是该技能的操作参考资料，用于辅助规划、执行与监督；如果与当前页面不一致，以实时截图为准。",
        }
    ]

    count = 0
    for p in paths:
        if count >= max_images:
            break
        url = _make_data_url(p)
        if not url:
            continue
        content.append({"type": "input_image", "image_url": url})
        count += 1

    if count == 0:
        logger.info(f"[SkillKB] 图片全部被过滤(可能>2MB或读取失败) skill={skill_name} dir={kb_dir}")
        return None
    logger.info(f"[SkillKB] 已加载 skill={skill_name} dir={kb_dir} images={count}")
    return {"role": "user", "content": content}
    logger.info(f"[SkillKB] 已加载(dir) skill={skill_name} dir={kb_dir} images={count}")


def _find_skill_kb_dir(skill_name: str, skill_info: Dict[str, Any]) -> Optional[Path]:
    kb_dir_cfg = skill_info.get("kb_dir")
    if isinstance(kb_dir_cfg, str) and kb_dir_cfg.strip():
        p = _resolve_skill_path(kb_dir_cfg)
        return p if p.exists() and p.is_dir() else None

    root = _find_vision_kb_root()
    if not root:
        return None
    candidates: List[str] = [skill_name]
    aliases = skill_info.get("aliases", [])
    if isinstance(aliases, list):
        candidates.extend(str(a) for a in aliases if a)
    normalized_candidates: List[str] = []
    for c in candidates:
        if not isinstance(c, str):
            continue
        if c and c not in normalized_candidates:
            normalized_candidates.append(c)
        lowered = c.lower()
        if lowered and lowered not in normalized_candidates:
            normalized_candidates.append(lowered)
    for name in normalized_candidates:
        p = root / name
        if p.exists() and p.is_dir():
            return p
    return None


def _count_available_kb_images(kb_dir: Optional[Path]) -> int:
    if not kb_dir or not kb_dir.exists() or not kb_dir.is_dir():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    try:
        return sum(1 for p in kb_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    except Exception:
        return 0


def _has_skill_kb(messages: List[Dict[str, Any]], skill_name: str) -> bool:
    marker = f"[SkillKB:{skill_name}]"
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str) and marker in content:
            return True
        if isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") in ("input_text", "text")
                    and isinstance(item.get("text"), str)
                    and marker in item["text"]
                ):
                    return True
    return False


def _inject_skill_kb(
    messages: List[Dict[str, Any]], kb_message: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not kb_message:
        return messages
    skill_name = None
    kb_content = kb_message.get("content")
    if isinstance(kb_content, list):
        for item in kb_content:
            if isinstance(item, dict) and item.get("type") == "input_text":
                t = item.get("text", "")
                if isinstance(t, str) and t.startswith("[SkillKB:") and "]" in t:
                    skill_name = t.split("]", 1)[0].replace("[SkillKB:", "")
                    break
    if not skill_name:
        return messages
    if _has_skill_kb(messages, skill_name):
        return messages
    logger.info(f"[SkillKB] 注入到上下文 skill={skill_name} images={_count_kb_images(kb_message)}")
    return [*messages, kb_message]


@register_agent(r"^plan-and-execute/.*", priority=100)
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
        self.skill_name: Optional[str] = None
        self.skill_instructions_text: str = ""
        self.skill_kb_message: Optional[Dict[str, Any]] = None
        self.load_skill_images: bool = False
        self._logged_plan_binding: bool = False

    async def _acompletion_json(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        valid_kwargs: Dict[str, Any],
    ) -> Any:
        try:
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    **valid_kwargs,
                )
            except Exception as e:
                if "response_format" not in str(e):
                    raise
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    **valid_kwargs,
                )
            content = response.choices[0].message.content
            return robust_json_loads(content)
        except Exception as e:
            logger.error(f"[Planner] json_call_failed: {e}")
            return None

    async def _call_skill_selector(
        self, messages: List[Dict[str, Any]], model: str, **kwargs
    ) -> Optional[str]:
        completion_messages = convert_responses_items_to_completion_messages(
            messages, allow_images_in_tool_results=True
        )

        skills = _load_skills()
        skills_path = _find_skills_json()
        logger.info(
            f"[Planner] skills_loaded={len(skills)} skills_path={str(skills_path) if skills_path else 'None'}"
        )
        skills_lines = []
        for name, info in skills.items():
            if not isinstance(info, dict):
                continue
            desc = info.get("description", "")
            if isinstance(desc, str) and desc.strip():
                skills_lines.append(f"- {name}: {desc.strip()}")
            else:
                skills_lines.append(f"- {name}")
        skills_block = "\n".join(skills_lines) if skills_lines else "- (none)"

        system_prompt = f"""You are selecting a skill for a Computer Use task.
Choose the best matching skill from the list. If none match, return null.

Available skills:
{skills_block}

Output a JSON object:
- skill_name: string | null
Example: {{"skill_name":"小红书"}} or {{"skill_name": null}}
"""
        selector_messages = [{"role": "system", "content": system_prompt}]
        selector_messages.extend(completion_messages)

        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["api_key", "api_base", "max_retries", "temperature", "top_p"]
        }

        raw = await self._acompletion_json(
            model=model, messages=selector_messages, valid_kwargs=valid_kwargs
        )
        if not isinstance(raw, dict):
            logger.info("[Planner] selected_skill=None")
            return None
        skill_name = raw.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name.strip():
            logger.info("[Planner] selected_skill=None")
            return None
        if skills and skill_name not in skills:
            logger.info(f"[Planner] selected_skill_invalid={skill_name}")
            return None
        logger.info(f"[Planner] selected_skill={skill_name}")
        return skill_name

    async def _call_load_images_decider(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        skill_name: str,
        sop_text: str,
        kb_dir: Optional[Path],
        images_available: int,
        **kwargs,
    ) -> bool:
        completion_messages = convert_responses_items_to_completion_messages(
            messages, allow_images_in_tool_results=True
        )

        system_prompt = f"""You are deciding whether to load skill-bound reference screenshots for planning.
Skill: {skill_name}
Images available: {images_available}
KB directory: {str(kb_dir) if kb_dir else "None"}

Skill SOP:
{sop_text}

If screenshots would likely help produce a more precise plan (e.g., identifying UI areas, typical layout, confirming which UI elements to target), answer true. Otherwise false.
Output a JSON object:
- load_skill_images: boolean
Example: {{"load_skill_images": true}}
"""
        decider_messages = [{"role": "system", "content": system_prompt}]
        decider_messages.extend(completion_messages)

        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["api_key", "api_base", "max_retries", "temperature", "top_p"]
        }
        raw = await self._acompletion_json(
            model=model, messages=decider_messages, valid_kwargs=valid_kwargs
        )
        if not isinstance(raw, dict):
            logger.info("[Planner] load_skill_images=false (invalid)")
            return False
        v = raw.get("load_skill_images", False)
        load = v is True
        logger.info(f"[Planner] load_skill_images={load} images_available={images_available}")
        return load

    async def _call_plan_generator(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        skill_name: Optional[str],
        sop_text: str,
        load_skill_images: bool,
        **kwargs,
    ) -> List[str]:
        completion_messages = convert_responses_items_to_completion_messages(
            messages, allow_images_in_tool_results=True
        )

        system_prompt = f"""You are a Planner Agent for a Computer Use task.
Your job is to produce a list of specific, simple UI steps that the Executor will carry out one by one.

Skill binding: {skill_name if skill_name else "None"}
Skill SOP (follow when applicable):
{sop_text if sop_text else "(none)"}

If reference screenshots are present in the context, use them to make the plan more precise by adding UI cues:
- Describe approximate location (e.g., top center, left sidebar, bottom right)
- Mention visual cues (icon shape, button label text, color theme) when reliable
- Do NOT use pixel coordinates
- Each step must be a single UI operation
- If the live screen differs, the Executor should adapt based on the latest screenshot

Output a JSON object:
- plan: string[]
Example:
{{"plan":[
  "点击页面顶部中间的搜索框（通常带放大镜图标）",
  "输入关键词“周也”并按回车（若有联想下拉，优先选第一个）"
]}}
"""
        planner_messages = [{"role": "system", "content": system_prompt}]
        planner_messages.extend(completion_messages)

        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["api_key", "api_base", "max_retries", "temperature", "top_p"]
        }
        raw = await self._acompletion_json(
            model=model, messages=planner_messages, valid_kwargs=valid_kwargs
        )
        if not isinstance(raw, dict):
            return ["Execute user request"]
        plan = raw.get("plan", ["Execute user request"])
        if not isinstance(plan, list) or not plan:
            return ["Execute user request"]
        return [str(s) for s in plan if str(s).strip()] or ["Execute user request"]

    async def _call_supervisor(
        self, messages: List[Dict[str, Any]], current_step: str, model: str, **kwargs
    ) -> Dict[str, Any]:
        # Get the recent history (e.g. last 5 messages to save tokens but keep the last action and its result)
        recent_messages = messages[-5:] if len(messages) >= 5 else messages
        completion_messages = convert_responses_items_to_completion_messages(
            recent_messages, allow_images_in_tool_results=True
        )

        sop_block = ""
        if self.skill_name or self.skill_instructions_text:
            sop_block = f"\nSkill: {self.skill_name if self.skill_name else 'None'}\nSOP:\n{self.skill_instructions_text}\n"

        system_prompt = f"""You are a Supervisor Agent.
Your job is to evaluate the result of the Executor's last action based on the screenshot and action history.
Current Step to accomplish: {current_step}
{sop_block}

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
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=supervisor_messages,
                    response_format={"type": "json_object"},
                    **valid_kwargs,
                )
            except Exception as e:
                if "response_format" not in str(e):
                    raise
                response = await litellm.acompletion(
                    model=model,
                    messages=supervisor_messages,
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
                has_text = any(
                    item.get("type") in ("text", "input_text")
                    for item in content
                    if isinstance(item, dict)
                )
                if has_text:
                    is_new_task = True

        # 1. Planner Phase
        if is_new_task or not self.plan:
            self._logged_plan_binding = False
            self.skill_name = None
            self.skill_instructions_text = ""
            self.skill_kb_message = None
            self.load_skill_images = False

            selected_skill = await self._call_skill_selector(messages, underlying_model, **kwargs)
            skills = _load_skills()
            skill_info = skills.get(selected_skill) if selected_skill else None
            if selected_skill and isinstance(skill_info, dict):
                self.skill_name = selected_skill
                logger.info(f"[SkillTool] auto_call get_skill_instructions skill={self.skill_name}")
                try:
                    tool = SkillTool()
                    sop = tool.call({"skill_name": self.skill_name})
                    self.skill_instructions_text = str(sop) if sop else ""
                except Exception as e:
                    logger.info(f"[SkillTool] failed skill={self.skill_name} error={e}")
                    self.skill_instructions_text = ""

                kb_dir = _find_skill_kb_dir(self.skill_name, skill_info)
                images_available = _count_available_kb_images(kb_dir)
                self.load_skill_images = await self._call_load_images_decider(
                    messages,
                    underlying_model,
                    skill_name=self.skill_name,
                    sop_text=self.skill_instructions_text,
                    kb_dir=kb_dir,
                    images_available=images_available,
                    **kwargs,
                )
                if self.load_skill_images:
                    self.skill_kb_message = _build_skill_kb_message(self.skill_name, skill_info)

            messages_for_planner = _inject_skill_kb(messages, self.skill_kb_message)
            self.plan = await self._call_plan_generator(
                messages_for_planner,
                underlying_model,
                skill_name=self.skill_name,
                sop_text=self.skill_instructions_text,
                load_skill_images=self.load_skill_images,
                **kwargs,
            )

            self.current_step_index = 0
            self.last_feedback = ""
            if not self._logged_plan_binding:
                logger.info(
                    f"[PlanAndExecute] bound_skill={self.skill_name if self.skill_name else 'None'} load_skill_images={self.load_skill_images} skill_images={_count_kb_images(self.skill_kb_message)}"
                )
                self._logged_plan_binding = True
            logger.info(f"Plan generated: {self.plan}")

        # 2. Supervisor Phase
        elif last_msg.get("type") == "computer_call_output" or (
            last_msg.get("role") == "user" and "input_image" in str(last_msg)
        ):
            if self.current_step_index < len(self.plan):
                current_step = self.plan[self.current_step_index]
                messages_for_supervisor = _inject_skill_kb(messages, self.skill_kb_message)
                eval_result = await self._call_supervisor(
                    messages_for_supervisor, current_step, underlying_model, **kwargs
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
        executor_messages = _inject_skill_kb(executor_messages, self.skill_kb_message)
        current_step_desc = self.plan[self.current_step_index]

        instruction = "You are the Executor Agent.\n"
        instruction += f"Overall Plan: {json.dumps(self.plan)}\n"
        if self.skill_name:
            instruction += f"Bound Skill: {self.skill_name}\n"
        if self.skill_instructions_text:
            instruction += f"Skill SOP:\n{self.skill_instructions_text}\n"
        instruction += (
            f"Current Step ({self.current_step_index + 1}/{len(self.plan)}): {current_step_desc}\n"
        )
        if self.last_feedback:
            instruction += f"Supervisor Feedback on last action: {self.last_feedback}\n"
        instruction += "Please take the next simple action using the provided tools to accomplish the current step."

        executor_messages.append({"role": "user", "content": instruction})

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
