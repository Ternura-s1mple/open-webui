from abc import ABC, abstractmethod
from langgraph.graph import StateGraph, START, END

class BaseAgent(ABC):
    def __init__(self, state_cls, config_schema=None):
        self.state_cls = state_cls
        self.config_schema = config_schema
        self.builder = StateGraph(state_cls, config_schema=config_schema)
        self._register_nodes()
        self._register_edges()

    def _register_nodes(self):
        self.builder.add_node("generate_query", self.generate_query)
        self.builder.add_node("web_research", self.web_research)
        self.builder.add_node("reflection", self.reflection)
        self.builder.add_node("finalize_answer", self.finalize_answer)

    def _register_edges(self):
        self.builder.add_edge(START, "generate_query")
        self.builder.add_conditional_edges(
            "generate_query", self.continue_to_web_research, ["web_research"]
        )
        self.builder.add_edge("web_research", "reflection")
        self.builder.add_conditional_edges(
            "reflection", self.evaluate_research, ["web_research", "finalize_answer"]
        )
        self.builder.add_edge("finalize_answer", END)

    @abstractmethod
    def setup_tools(self):
        """抽象方法：设置agent可用工具"""
        pass

    # 以下方法可根据需要在子类中重写
    def generate_query(self, state, config):
        raise NotImplementedError
    def continue_to_web_research(self, state):
        raise NotImplementedError
    def web_research(self, state, config):
        raise NotImplementedError
    def reflection(self, state, config):
        raise NotImplementedError
    def evaluate_research(self, state, config):
        raise NotImplementedError
    def finalize_answer(self, state, config):
        raise NotImplementedError

    def compile(self, name):
        return self.builder.compile(name=name) 