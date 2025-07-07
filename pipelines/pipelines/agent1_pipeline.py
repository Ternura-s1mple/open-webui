# -*- coding: utf-8 -*-
"""
OpenWebUI Pipeline: Agent1
本文件作为 OpenWebUI 的 pipeline 入口，复用 agent1_package/agent/agent1.py 中的 Agent1 实例。
"""
import sys
import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field # 导入 Field
import uuid # 导入uuid模块用于生成随机ID，以防thread_id缺失
import traceback # 导入 traceback 用于打印详细错误堆栈

# 保证 agent1_package/agent 和 agent1_core 能被正确导入
# 假设此 pipeline.py 文件位于 Open WebUI 管道目录的某个子目录中
# 例如：/app/pipelines/my_agent_pipeline/pipeline.py
# 那么 agent1_package 应该在 /app/agent1_package
# 这里的路径需要根据您的实际部署结构进行调整
# 如果 agent1_package 在 /app/agent1_package，那么 __file__ 的相对路径可能不同
# 建议在 Dockerfile 中设置 PYTHONPATH 或直接 COPY 到 /app
# 例如：COPY ./agent1_package /app/agent1_package
# 然后在 Python 代码中直接 import agent1_package.agent.agent1
# 或者，如果 agent1_package 就在当前目录的上一级，可以这样：
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) # 如果 agent1_package 在当前目录的上一级

# 假设 agent1_package 已经被正确地复制到 Docker 容器的 Python 路径中
# 例如，在 Dockerfile 中：
# COPY ./agent1_package /app/agent1_package
# ENV PYTHONPATH=/app:$PYTHONPATH
# 那么这里可以直接导入
from agent1_package.agent.agent1 import Agent1
from agent1_package.agent1_core.state.state import OverallState
from agent1_package.agent1_core.config.configuration import Configuration

class Pipeline:
    # 定义 Valves 类，用于从 Open WebUI 的配置中获取参数
    # 这里我们不再直接从 os.getenv 获取，而是让 Agent1 内部处理
    # 或者，如果 Agent1 的 Configuration 需要这些值，可以在这里传递
    class Valves(BaseModel):
        # GEMINI_API_KEY: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "your_gemini_api_key"))
        # QUERY_GENERATOR_MODEL: str = "gemini-2.0-flash"
        # REFLECTION_MODEL: str = "gemini-2.5-flash"
        # ANSWER_MODEL: str = "gemini-2.5-pro-preview-05-06"
        # NUMBER_OF_INITIAL_QUERIES: int = 3
        # MAX_RESEARCH_LOOPS: int = 3
        # 由于 Agent1 内部会从 Configuration 获取这些，这里可以简化
        # 或者，如果 Open WebUI 的 UI 允许配置这些，Valves 可以定义它们
        pass # 如果不需要从 Open WebUI UI 配置额外参数，Valves 可以为空

    def __init__(self):
        # 初始化 Valves（如果定义了参数）
        self.valves = self.Valves()
        self.name = "Agent1 Research Assistant" # 管道名称

        # 实例化 Agent1，并复用其编译好的 LangGraph 应用
        # Agent1 的构造函数需要 OverallState 和 Configuration schema
        self.agent_instance = Agent1(
            state_cls=OverallState,
            config_schema=Configuration
        )
        # 编译 Agent1 的 LangGraph 工作流
        self.app = self.agent_instance.compile(name="pro-search-agent")

    # 修正 pipe 方法签名，使其符合 Open WebUI 管道接口
    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """
        OpenWebUI 标准 pipeline 接口。
        接收用户消息、模型ID、消息历史和请求体，传递给 LangGraph 应用。
        """
        try:
            # 1. 获取或生成 thread_id，用于 LangGraph 的会话管理
            # Open WebUI 通常会在 body 中传递会话相关的 ID
            thread_id = body.get("thread_id") or body.get("conversation_id") or str(uuid.uuid4())
            print(f"DEBUG: Using thread_id: {thread_id} for LangGraph session.")
            print(f"DEBUG: Raw body received: {body}")

            # 2. 准备 LangGraph 的初始状态
            # messages 是 OverallState 的一部分
            initial_state = OverallState(messages=messages)

            # 3. 准备 LangGraph 的配置字典
            # 这些配置会通过 LangGraph 传递到 Agent1 的节点函数中
            # 这里的配置应该与 Agent1 的 Configuration schema 匹配
            langgraph_config = {
                "configurable": {
                    "thread_id": thread_id, # 这是 LangGraph Checkpointer 识别会话的关键

                    # 修正默认值获取方式
                    "query_generator_model": Configuration.model_fields['query_generator_model'].default,
                    "reflection_model": Configuration.model_fields['reflection_model'].default,
                    "answer_model": Configuration.model_fields['answer_model'].default,
                    "number_of_initial_queries": Configuration.model_fields['number_of_initial_queries'].default,
                    "max_research_loops": Configuration.model_fields['max_research_loops'].default,
                    # 如果 Agent1 的 Configuration 还需要 GEMINI_API_KEY，也应该在这里传递
                    # "gemini_api_key": self.agent_instance.config_schema.GEMINI_API_KEY.default,
                    # 或者从 self.valves 获取，如果 Open WebUI UI 允许配置
                }
            }

            # 检查 body 中是否有覆盖配置，如果有，则更新 langgraph_config
            # 假设 Open WebUI 会在 body.get("config") 中传递额外的可配置参数
            # 这是一个常见的模式，但具体取决于 Open WebUI 如何设计其管道配置传递
            if "config" in body and isinstance(body["config"], dict):
                for key, value in body["config"].items():
                    if key in langgraph_config["configurable"]:
                        langgraph_config["configurable"][key] = value
                print(f"DEBUG: Updated langgraph_config with body config: {langgraph_config}")


            # 4. 执行 LangGraph 工作流
            # self.app 是 Agent1 编译好的 LangGraph 应用
            result_state = self.app.invoke(initial_state, langgraph_config)

            # 5. 从 LangGraph 返回的最终状态中提取答案
            # LangGraph invoke 返回的是最终状态对象，从中提取 messages
            if result_state and "messages" in result_state and result_state["messages"]:
                final_message = result_state["messages"][-1]
                # Open WebUI 管道通常期望返回字符串或生成器
                return final_message.content
            else:
                return "抱歉，无法生成答案。"

        except Exception as e:
            print(f"ERROR: Unexpected error in pipe: {e}")
            traceback.print_exc() # 打印完整的堆栈跟踪，方便调试
            return f"Unexpected error: {e}"

# 兼容自动加载
pipeline = Pipeline()