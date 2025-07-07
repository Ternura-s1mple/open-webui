from abc import ABC, abstractmethod
from typing import Any, Literal
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.tongyi import ChatTongyi
import os

class ModelAdapter(ABC):
    """模型适配器基类"""
    
    @abstractmethod
    def create_chat_model(
        self,
        model_name: str,
        temperature: float = 0,
        max_retries: int = 2,
        **kwargs
    ) -> BaseChatModel:
        """创建聊天模型实例"""
        pass

    @abstractmethod
    def create_structured_output(
        self,
        model: BaseChatModel,
        output_schema: Any
    ) -> Any:
        """创建结构化输出模型"""
        pass

class GeminiAdapter(ModelAdapter):
    """Gemini模型适配器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_chat_model(
        self,
        model_name: str,
        temperature: float = 0,
        max_retries: int = 2,
        **kwargs
    ) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
            api_key=self.api_key,
            **kwargs
        )

    def create_structured_output(
        self,
        model: BaseChatModel,
        output_schema: Any
    ) -> Any:
        return model.with_structured_output(output_schema)

class QianwenAdapter(ModelAdapter):
    """千问模型适配器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_chat_model(
        self,
        model_name: str,
        temperature: float = 0,
        max_retries: int = 2,
        **kwargs
    ) -> BaseChatModel:
        return ChatTongyi(
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
            dashscope_api_key=self.api_key,
            **kwargs
        )

    def create_structured_output(
        self,
        model: BaseChatModel,
        output_schema: Any
    ) -> Any:
        print(f"=== 千问使用PydanticOutputParser ===")
        
        from langchain.output_parsers import PydanticOutputParser
        
        # 创建解析器
        parser = PydanticOutputParser(pydantic_object=output_schema)
        
        # 创建包装器
        class QianwenStructuredWrapper:
            def __init__(self, model, parser):
                self.model = model
                self.parser = parser
            
            def invoke(self, prompt):
                print("使用PydanticOutputParser解析千问输出")
                
                # 添加格式说明到提示词
                format_instructions = self.parser.get_format_instructions()
                full_prompt = f"{prompt}\n\n{format_instructions}"
                
                # 调用模型
                response = self.model.invoke(full_prompt)
                print(f"千问原始响应: {response.content}")
                
                # 解析输出
                try:
                    result = self.parser.parse(response.content)
                    print(f"解析成功: {result}")
                    return result
                except Exception as e:
                    print(f"解析失败: {e}")
                    # 尝试手动解析JSON
                    import re
                    import json
                    
                    # 查找JSON部分
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        try:
                            parsed_data = json.loads(json_str)
                            return output_schema(**parsed_data)
                        except Exception as json_e:
                            print(f"JSON解析也失败: {json_e}")
                            raise
                    else:
                        raise ValueError("无法解析响应")
        
        return QianwenStructuredWrapper(model, parser)

# 在这里指定使用哪个系列的模型
MODEL_SERIES: Literal["gemini", "qianwen"] = "qianwen"  # 修改这里来切换模型系列

def get_model_adapter(model_name: str) -> ModelAdapter:
    """根据指定的模型系列创建对应的适配器"""
    if MODEL_SERIES == "gemini":
        return GeminiAdapter(api_key=os.getenv("GEMINI_API_KEY"))
    elif MODEL_SERIES == "qianwen":
        return QianwenAdapter(api_key=os.getenv("DASHSCOPE_API_KEY"))
    else:
        raise ValueError(f"Unsupported model series: {MODEL_SERIES}") 