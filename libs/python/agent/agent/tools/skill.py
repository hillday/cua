"""
Skill tool for the agent.
Allows the agent to load and retrieve step-by-step instructions for specific predefined skills.
"""

import json
import os
from pathlib import Path
from typing import Dict, Union

from .base import BaseTool, register_tool


def load_skills() -> Dict[str, dict]:
    """
    Load skills from skills.json file in the project root.
    """
    # 尝试多种可能的路径
    potential_paths = [
        Path(os.getcwd()) / "skills.json", # 当前目录
        Path(os.getcwd()).parent / "skills.json", # 上级目录
        Path(__file__).resolve().parents[4] / "skills.json", # 项目根目录 (根据文件深度计算)
    ]
    
    # 额外向上查找 5 级
    current_dir = Path(__file__).resolve().parent
    for _ in range(5):
        potential_paths.append(current_dir / "skills.json")
        current_dir = current_dir.parent
            
    for skills_path in potential_paths:
        if skills_path.exists():
            try:
                with open(skills_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading skills from {skills_path}: {e}")
            
    return {}


@register_tool("get_skill_instructions")
class SkillTool(BaseTool):
    """
    A tool that allows the agent to retrieve predefined step-by-step instructions for a skill.
    """

    name = "get_skill_instructions"

    def __init__(self, cfg=None):
        self.skills = load_skills()
        super().__init__(cfg)

    @property
    def description(self) -> str:
        if not self.skills:
            return "Retrieve step-by-step instructions for a specific skill by name. (Currently no skills loaded)"
        
        skill_descriptions = []
        for name, info in self.skills.items():
            desc = info.get("description", "No description")
            aliases = info.get("aliases", [])
            alias_str = f" (别名: {', '.join(aliases)})" if aliases else ""
            skill_descriptions.append(f"- {name}{alias_str}: {desc}")
            
        return (
            "IMPORTANT: MANDATORY TOOL for complex tasks. "
            "You MUST first call this tool to retrieve the verified SOP (Standard Operating Procedure) instructions "
            "before performing any actions for the following tasks: "
            f"\n" + "\n".join(skill_descriptions) + 
            "\nAlways follow the retrieved steps strictly to ensure task success."
        )

    @property
    def parameters(self) -> dict:
        available_skills = list(self.skills.keys()) if self.skills else ["none"]

        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": f"The name of the skill to retrieve. Available skills: {', '.join(available_skills)}",
                }
            },
            "required": ["skill_name"],
        }

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        params_dict = self._verify_json_format_args(params)
        skill_name = params_dict.get("skill_name")
        
        # Reload skills in case the file was updated
        self.skills = load_skills()
        
        # 1. Try exact match
        skill_info = self.skills.get(skill_name)
        
        # 2. Try alias match if exact match fails
        if not skill_info:
            for name, info in self.skills.items():
                aliases = info.get("aliases", [])
                if skill_name in aliases or skill_name.lower() in [a.lower() for a in aliases]:
                    skill_info = info
                    skill_name = name
                    break
        
        # 3. Try fuzzy match (if skill_name is part of the key or vice-versa)
        if not skill_info:
            for name, info in self.skills.items():
                if skill_name in name or name in skill_name:
                    skill_info = info
                    skill_name = name
                    break

        if not skill_info:
            available = ", ".join(self.skills.keys()) if self.skills else "None"
            return f"Error: Skill '{skill_name}' not found. Available skills are: {available}"
            
        instructions = skill_info.get("instructions", [])

        if not instructions:
            return f"Error: Skill '{skill_name}' has no instructions defined."

        if isinstance(instructions, list):
            instructions_text = "\n".join(instructions)
        else:
            instructions_text = str(instructions)

        return (
            f"=== Instructions for skill '{skill_name}' ===\n"
            f"Description: {skill_info.get('description', 'N/A')}\n\n"
            f"Steps to execute:\n"
            f"{instructions_text}\n\n"
            f"Please follow these steps carefully to complete the task."
        )
