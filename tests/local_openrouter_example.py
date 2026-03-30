import asyncio
import logging
import os

from agent import ComputerAgent
from agent.tools.skill import SkillTool
from computer import Computer

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("agent_run.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def main():
    # 确保设置了 OpenRouter 的 API Key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("警告: 未检测到 OPENROUTER_API_KEY 环境变量。")
        print("如果代码执行失败，请在运行前执行: export OPENROUTER_API_KEY='你的key'")

    # 1. 初始化 Computer 对象
    # 关键参数:
    # - use_host_computer_server=True: 这告诉 CUA 不要去启动 Docker/VM，而是直接连接本地宿主机
    # - os_type="windows": 告诉系统当前是 Windows (根据你实际操作系统修改，如 macos, linux)
    computer = Computer(os_type="windows", use_host_computer_server=True, verbosity=logging.DEBUG)

    # 2. 启动本地计算机服务器并进入上下文
    print("正在启动本地控制服务...")
    await computer.__aenter__()

    try:
        # 3. 初始化大模型代理

        # 选项 A: 使用 OpenRouter (通过 OpenAI 适配器强制启用 Native Tool Call)
        """
        agent = ComputerAgent(
            # 将前缀改为 openai/ 强制触发 openai.py 适配器，跳过 GenericVLM 的 XML 逻辑
            model="openai/qwen/qwen3.5-122b-a10b",
            tools=[computer, SkillTool()],
            instructions="优先检查是否有可用的 skill 能够辅助完成任务。如果任务涉及小红书、Github 等已定义技能的操作，必须先调用 get_skill_instructions 获取 SOP 步骤并严格执行。如果你发现任务可以通过预定义的 skill 完成，请先获取步骤，并在后续思考中明确你正在执行哪一个步骤。",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            # 显式指定 OpenRouter 的 API 基地址
            api_base="https://openrouter.ai/api/v1",
        )
        """

        # 选项 B: 使用 火山引擎 (已注释)

        agent = ComputerAgent(
            # 在模型名称前加上 plan-and-execute/ 前缀，即可启用我们刚刚创建的 Plan-and-Execute 架构
            model="plan-and-execute/openai/doubao-seed-2-0-lite-260215",
            tools=[computer, SkillTool()],
            api_key=os.getenv("VOLCENGINE_API_KEY"),
            screenshot_delay=1.0,
            api_base="https://ark.cn-beijing.volces.com/api/v3",
        )

        # 4. 给定任务并运行
        task = (
            "打开chrome浏览器,访问小红书，搜索周也的最近三天5个热门帖子，给帖子的第一个评论点赞。"
        )
        history = [{"role": "user", "content": task}]

        logger.info(f"\n开始执行任务: {task}")
        logger.warning(
            "警告：此时 Agent 将直接控制你的鼠标和键盘！请随时准备移动鼠标夺回控制权。\n"
        )

        # 等待几秒钟让你准备好
        await asyncio.sleep(3)

        async for result in agent.run(history):
            # 将输出加入历史记录中，维持上下文循环
            history.extend(result.get("output", []))

            for item in result.get("output", []):
                if item.get("type") == "message" and item.get("role") == "assistant":
                    content = item.get("content", [])
                    for part in content:
                        if part.get("text"):
                            logger.info(f"\n[Agent 思考/回复]: {part['text']}")
                elif item.get("type") == "computer_call":
                    action = item.get("action", {})
                    action_type = action.get("type", "")
                    if action_type:
                        logger.info(f"\n🛠️ [执行本地操作]: {action_type} - {action}")

    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
    finally:
        # 5. 确保清理和断开连接
        logger.info("\n任务结束，正在断开连接...")
        await computer.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
