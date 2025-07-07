"""
Agent1 Core Package

这个包包含了Agent1研究助手的核心功能模块，包括：
- state: 状态管理
- config: 配置管理
- prompts: 提示词模板
- tools: 工具函数
- models: 模型适配器
- schemas: 数据模式定义
"""

__version__ = "1.0.0"
__author__ = "Agent1 Team"

# 导出主要模块
from .state.state import OverallState
from .config.configuration import Configuration
from .models.model_adapter import get_model_adapter, MODEL_SERIES
from .schemas.tools_and_schemas import SearchQueryList, Reflection

__all__ = [
    "OverallState",
    "Configuration", 
    "get_model_adapter",
    "MODEL_SERIES",
    "SearchQueryList",
    "Reflection"
] 