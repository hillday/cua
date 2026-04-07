"""
Plan-and-Execute agent loop implementation.
"""

from __future__ import annotations

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


def _count_kb_images(kb_message: Optional[Dict[str, Any]]) -> int:
    if not kb_message:
        return 0
    content = kb_message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1 for item in content if isinstance(item, dict) and item.get("type") == "input_image"
    )


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
    kb_content = kb_message.get("content")
    skill_name = None
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
    logger.info(f"[SkillKB] inject skill={skill_name} images={_count_kb_images(kb_message)}")
    return [*messages, kb_message]


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
            logger.info(f"[SkillKB] kb_images configured but none loadable skill={skill_name}")
            return None
        logger.info(f"[SkillKB] loaded(kb_images) skill={skill_name} images={loaded}")
        return {"role": "user", "content": content}

    kb_dir_cfg = skill_info.get("kb_dir")
    if isinstance(kb_dir_cfg, str) and kb_dir_cfg.strip():
        kb_dir = _resolve_skill_path(kb_dir_cfg)
        if not kb_dir.exists() or not kb_dir.is_dir():
            logger.info(f"[SkillKB] kb_dir not found skill={skill_name} kb_dir={kb_dir}")
            return None
    else:
        root = _find_vision_kb_root()
        if not root:
            logger.info(f"[SkillKB] no vision_kb root skill={skill_name}")
            return None
        candidates: List[str] = [skill_name]
        aliases = skill_info.get("aliases", [])
        if isinstance(aliases, list):
            candidates.extend(str(a) for a in aliases if a)
        normalized: List[str] = []
        for c in candidates:
            if not isinstance(c, str):
                continue
            if c and c not in normalized:
                normalized.append(c)
            lowered = c.lower()
            if lowered and lowered not in normalized:
                normalized.append(lowered)
        kb_dir = None
        for name in normalized:
            p = root / name
            if p.exists() and p.is_dir():
                kb_dir = p
                break
        if not kb_dir:
            logger.info(f"[SkillKB] no matching dir skill={skill_name}")
            return None

    max_images = int(skill_info.get("kb_max_images") or os.getenv("CUA_VISION_KB_MAX_IMAGES", "6"))
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths = [p for p in sorted(kb_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        logger.info(f"[SkillKB] empty dir skill={skill_name} dir={kb_dir}")
        return None

    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": f"[SkillKB:{skill_name}] 以下截图是该技能的操作参考资料，用于辅助规划、执行与监督；如果与当前页面不一致，以实时截图为准。",
        }
    ]
    loaded = 0
    for p in paths:
        if loaded >= max_images:
            break
        url = _make_data_url(p)
        if not url:
            continue
        content.append({"type": "input_image", "image_url": url})
        loaded += 1

    if loaded == 0:
        logger.info(f"[SkillKB] images filtered skill={skill_name} dir={kb_dir}")
        return None
    logger.info(f"[SkillKB] loaded(dir) skill={skill_name} dir={kb_dir} images={loaded}")
    return {"role": "user", "content": content}


def _count_available_images(skill_name: str, skill_info: Dict[str, Any]) -> int:
    kb_images = skill_info.get("kb_images")
    if isinstance(kb_images, list) and kb_images:
        return sum(1 for v in kb_images if isinstance(v, str) and _resolve_skill_path(v).exists())
    kb_dir_cfg = skill_info.get("kb_dir")
    kb_dir = None
    if isinstance(kb_dir_cfg, str) and kb_dir_cfg.strip():
        p = _resolve_skill_path(kb_dir_cfg)
        kb_dir = p if p.exists() and p.is_dir() else None
    if not kb_dir:
        root = _find_vision_kb_root()
        if not root:
            return 0
        for name in [skill_name, *(skill_info.get("aliases") or [])]:
            if not isinstance(name, str) or not name:
                continue
            p = root / name
            if p.exists() and p.is_dir():
                kb_dir = p
                break
            p2 = root / name.lower()
            if p2.exists() and p2.is_dir():
                kb_dir = p2
                break
    if not kb_dir:
        return 0
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    try:
        return sum(1 for p in kb_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    except Exception:
        return 0


@register_agent(r"^plan-and-execute/.*", priority=100)
class PlanAndExecuteConfig(AsyncAgentConfig):
    def __init__(self):
        self.plan: List[str] = []
        self.current_step_index: int = 0
        self.last_feedback: str = ""
        self.executor_loop: Optional[AsyncAgentConfig] = None
        self.skill_name: Optional[str] = None
        self.skill_sop: str = ""
        self.load_skill_images: bool = False
        self.skill_kb_message: Optional[Dict[str, Any]] = None

    def _valid_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v
            for k, v in kwargs.items()
            if k in ["api_key", "api_base", "max_retries", "temperature", "top_p"]
        }

    async def _acompletion_json(
        self, *, model: str, messages: List[Dict[str, Any]], valid_kwargs: Dict[str, Any]
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
                response = await litellm.acompletion(model=model, messages=messages, **valid_kwargs)
            content = response.choices[0].message.content
            return robust_json_loads(content)
        except Exception as e:
            logger.error(f"[PlanAndExecute] acompletion_json failed: {e}")
            return None

    async def _select_skill(
        self, messages: List[Dict[str, Any]], model: str, **kwargs
    ) -> Optional[str]:
        skills = _load_skills()
        skills_path = _find_skills_json()
        logger.info(
            f"[Planner] skills_loaded={len(skills)} skills_path={str(skills_path) if skills_path else 'None'}"
        )
        if not skills:
            return None

        lines = []
        for name, info in skills.items():
            if not isinstance(info, dict):
                continue
            desc = info.get("description", "")
            if isinstance(desc, str) and desc.strip():
                lines.append(f"- {name}: {desc.strip()}")
            else:
                lines.append(f"- {name}")
        skills_block = "\n".join(lines) if lines else "- (none)"

        completion_messages = convert_responses_items_to_completion_messages(
            messages, allow_images_in_tool_results=True
        )
        system_prompt = f"""Choose the best matching skill for the user's request. If none match, return null.

Available skills:
{skills_block}

Output a JSON object:
{{"skill_name": "..."}} or {{"skill_name": null}}
"""
        selector_messages = [{"role": "system", "content": system_prompt}, *completion_messages]
        raw = await self._acompletion_json(
            model=model, messages=selector_messages, valid_kwargs=self._valid_kwargs(kwargs)
        )
        if not isinstance(raw, dict):
            return None
        skill_name = raw.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name.strip():
            return None
        if skill_name not in skills:
            return None
        logger.info(f"[Planner] selected_skill={skill_name}")
        return skill_name

    async def _decide_load_images(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        skill_name: str,
        sop_text: str,
        images_available: int,
        **kwargs,
    ) -> bool:
        if images_available <= 0:
            return False
        completion_messages = convert_responses_items_to_completion_messages(
            messages, allow_images_in_tool_results=True
        )
        system_prompt = f"""Decide whether to load skill-bound reference screenshots as additional context.
Skill: {skill_name}
Images available: {images_available}

SOP:
{sop_text}

Return true only if reference screenshots would likely help produce a more precise, stable plan.
Output JSON: {{"load_skill_images": true|false}}
"""
        decider_messages = [{"role": "system", "content": system_prompt}, *completion_messages]
        raw = await self._acompletion_json(
            model=model, messages=decider_messages, valid_kwargs=self._valid_kwargs(kwargs)
        )
        if not isinstance(raw, dict):
            return False
        v = raw.get("load_skill_images", False)
        load = v is True
        logger.info(f"[Planner] load_skill_images={load} images_available={images_available}")
        return load

    async def _generate_plan(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        skill_name: Optional[str],
        sop_text: str,
        **kwargs,
    ) -> List[str]:
        completion_messages = convert_responses_items_to_completion_messages(
            messages, allow_images_in_tool_results=True
        )
        system_prompt = f"""You are a Planner Agent for a Computer Use task.
Produce a list of specific, simple UI steps that the Executor will execute one by one.

Skill binding: {skill_name if skill_name else "None"}
SOP (follow when applicable):
{sop_text if sop_text else "(none)"}

If reference screenshots are present, use them to add execution-friendly UI cues:
- approximate location (e.g., top center, left sidebar, bottom right)
- visible cues (button label text, icon shape, color theme) when reliable
- do NOT use pixel coordinates

Output JSON: {{"plan": ["...", "..."]}}
"""
        planner_messages = [{"role": "system", "content": system_prompt}, *completion_messages]
        raw = await self._acompletion_json(
            model=model, messages=planner_messages, valid_kwargs=self._valid_kwargs(kwargs)
        )
        if not isinstance(raw, dict):
            return ["Execute user request"]
        plan = raw.get("plan", ["Execute user request"])
        if not isinstance(plan, list) or not plan:
            return ["Execute user request"]
        cleaned = [str(s).strip() for s in plan if str(s).strip()]
        return cleaned or ["Execute user request"]

    async def _call_supervisor(
        self, messages: List[Dict[str, Any]], current_step: str, model: str, **kwargs
    ) -> Dict[str, Any]:
        recent_messages = messages[-6:] if len(messages) >= 6 else messages
        completion_messages = convert_responses_items_to_completion_messages(
            recent_messages, allow_images_in_tool_results=True
        )
        sop_block = ""
        if self.skill_name or self.skill_sop:
            sop_block = f"\nSkill: {self.skill_name if self.skill_name else 'None'}\nSOP:\n{self.skill_sop}\n"
        system_prompt = f"""You are a Supervisor Agent.
Evaluate the result of the Executor's last action based on the screenshot and action history.
Current step: {current_step}
{sop_block}

Output JSON:
- status: one of action_success, action_failure, step_complete, task_complete
- feedback: brief guidance for the Executor
"""
        supervisor_messages = [{"role": "system", "content": system_prompt}, *completion_messages]
        raw = await self._acompletion_json(
            model=model, messages=supervisor_messages, valid_kwargs=self._valid_kwargs(kwargs)
        )
        return raw if isinstance(raw, dict) else {"status": "action_success", "feedback": ""}

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
            config_info = find_agent_config(underlying_model) or find_agent_config("openai/gpt-4o")
            self.executor_loop = config_info.agent_class()

        last_msg = messages[-1] if messages else {}
        is_new_task = False
        if isinstance(last_msg, dict) and last_msg.get("role") == "user":
            content = last_msg.get("content")
            if isinstance(content, str):
                is_new_task = True
            elif isinstance(content, list):
                has_text = any(
                    isinstance(it, dict) and it.get("type") in ("text", "input_text")
                    for it in content
                )
                is_new_task = has_text

        if is_new_task or not self.plan:
            self.plan = []
            self.current_step_index = 0
            self.last_feedback = ""
            self.skill_name = None
            self.skill_sop = ""
            self.load_skill_images = False
            self.skill_kb_message = None

            selected = await self._select_skill(messages, underlying_model, **kwargs)
            skills = _load_skills()
            skill_info = skills.get(selected) if selected else None
            if selected and isinstance(skill_info, dict):
                self.skill_name = selected
                instructions = skill_info.get("instructions", [])
                if isinstance(instructions, list):
                    self.skill_sop = "\n".join(str(s) for s in instructions if s)
                else:
                    self.skill_sop = str(instructions) if instructions else ""
                images_available = _count_available_images(selected, skill_info)
                self.load_skill_images = await self._decide_load_images(
                    messages,
                    underlying_model,
                    skill_name=selected,
                    sop_text=self.skill_sop,
                    images_available=images_available,
                    **kwargs,
                )
                if self.load_skill_images:
                    self.skill_kb_message = _build_skill_kb_message(selected, skill_info)

            plan_messages = _inject_skill_kb(messages, self.skill_kb_message)
            self.plan = await self._generate_plan(
                plan_messages,
                underlying_model,
                skill_name=self.skill_name,
                sop_text=self.skill_sop,
                **kwargs,
            )
            logger.info(
                f"[PlanAndExecute] bound_skill={self.skill_name if self.skill_name else 'None'} load_skill_images={self.load_skill_images} skill_images={_count_kb_images(self.skill_kb_message)} plan_len={len(self.plan)}"
            )

        if (
            isinstance(last_msg, dict)
            and last_msg.get("type") == "computer_call_output"
            and self.current_step_index < len(self.plan)
        ):
            current_step = self.plan[self.current_step_index]
            supervisor_messages = _inject_skill_kb(messages, self.skill_kb_message)
            eval_result = await self._call_supervisor(
                supervisor_messages, current_step, underlying_model, **kwargs
            )
            self.last_feedback = str(eval_result.get("feedback", "")).strip()
            status = eval_result.get("status")
            logger.info(
                f"[Supervisor] status={status} step={self.current_step_index+1}/{len(self.plan)}"
            )
            if status == "step_complete":
                self.current_step_index += 1
                self.last_feedback = ""
            if status == "task_complete" or self.current_step_index >= len(self.plan):
                return {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Task completed."}],
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

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

        executor_messages = _inject_skill_kb(messages.copy(), self.skill_kb_message)
        current_step_desc = self.plan[self.current_step_index]
        instruction = "You are the Executor Agent.\n"
        instruction += f"Overall Plan: {json.dumps(self.plan, ensure_ascii=False)}\n"
        instruction += (
            f"Current Step ({self.current_step_index + 1}/{len(self.plan)}): {current_step_desc}\n"
        )
        if self.skill_name:
            instruction += f"Bound Skill: {self.skill_name}\n"
        if self.skill_sop:
            instruction += f"SOP:\n{self.skill_sop}\n"
        if self.last_feedback:
            instruction += f"Supervisor Feedback: {self.last_feedback}\n"
        instruction += (
            "Use the provided tools to take a screenshot if needed and perform the next simple UI action. "
            "Use approximate location/visual cues (not pixel coordinates) when describing what to click."
        )
        executor_messages.append({"role": "user", "content": instruction})

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
