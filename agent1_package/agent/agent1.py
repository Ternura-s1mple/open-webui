import os
from langchain_core.messages import AIMessage
from langgraph.types import Send
from google.genai import Client
from agent1_core.state.state import OverallState, QueryGenerationState, ReflectionState, WebSearchState
from agent1_core.config.configuration import Configuration
from agent1_core.prompts.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent1_core.tools.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from agent1_core.models.model_adapter import get_model_adapter, MODEL_SERIES
from agent1_core.schemas.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from agent.baseagent import BaseAgent
from agent1_package.agent.local_rag_agent import LocalRAGAgent
import glob

load_dotenv()
if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

class Agent1(BaseAgent):
    def setup_tools(self):
        # 当前流程暂时不变，后续可扩展
        pass

    def generate_query(self, state, config):
        configurable = Configuration.from_runnable_config(config)
        if state.get("initial_search_query_count") is None:
            state["initial_search_query_count"] = configurable.number_of_initial_queries
        model_adapter = get_model_adapter(configurable.query_generator_model)
        llm = model_adapter.create_chat_model(
            model_name=configurable.query_generator_model,
            temperature=1.0,
            max_retries=2,
        )
        structured_llm = model_adapter.create_structured_output(llm, SearchQueryList)
        current_date = get_current_date()
        formatted_prompt = query_writer_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            number_queries=state["initial_search_query_count"],
        )
        try:
            result = structured_llm.invoke(formatted_prompt)
        except Exception:
            result = None
        if result is None:
            return {"query_list": ["default search query"]}
        return {"query_list": result.query}

    def continue_to_web_research(self, state):
        return [
            Send("web_research", {"search_query": search_query, "id": int(idx)})
            for idx, search_query in enumerate(state["query_list"])
        ]

    def web_research(self, state, config):
        # 检查向量库是否存在
        base_dir = os.path.dirname(os.path.abspath(__file__))
        vector_db_path = os.path.join(base_dir, "vectorstore/faiss_index_local_txt")
        local_pdf_folder = os.path.join(base_dir, "localpdf")
        # 检查PDF文件
        txt_files = glob.glob(os.path.join(local_pdf_folder, "*.txt"))
        if not os.path.exists(vector_db_path) or not txt_files:
            fake_result = "未找到向量库或TXT文件，请先上传TXT并完成向量化。"
            return {
                "sources_gathered": [],
                "search_query": [state["search_query"]],
                "web_research_result": [fake_result],
            }
        try:
            # 只检索TXT
            rag_agent = LocalRAGAgent(local_txt_folder=local_pdf_folder, vector_db_path=vector_db_path)
            answer, context = rag_agent.ask(state["search_query"])
            # 修正：将Document对象转为dict，避免后续subscriptable错误
            context_list = []
            for doc in context:
                context_list.append({
                    "page_content": getattr(doc, "page_content", ""),
                    "metadata": getattr(doc, "metadata", {})
                })
            return {
                "sources_gathered": context_list,
                "search_query": [state["search_query"]],
                "web_research_result": [answer],
            }
        except Exception as e:
            return {
                "sources_gathered": [],
                "search_query": [state["search_query"]],
                "web_research_result": [f"本地检索异常: {e}"],
            }

    def reflection(self, state, config):
        configurable = Configuration.from_runnable_config(config)
        state["research_loop_count"] = state.get("research_loop_count", 0) + 1
        reasoning_model = state.get("reasoning_model") or configurable.reflection_model
        model_adapter = get_model_adapter(reasoning_model)
        current_date = get_current_date()
        formatted_prompt = reflection_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            summaries="\n\n---\n\n".join(state["web_research_result"]),
        )
        llm = model_adapter.create_chat_model(
            model_name=reasoning_model,
            temperature=1.0,
            max_retries=2,
        )
        result = model_adapter.create_structured_output(llm, Reflection).invoke(formatted_prompt)
        return {
            "is_sufficient": result.is_sufficient,
            "knowledge_gap": result.knowledge_gap,
            "follow_up_queries": result.follow_up_queries,
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }

    def evaluate_research(self, state, config):
        configurable = Configuration.from_runnable_config(config)
        max_research_loops = (
            state.get("max_research_loops")
            if state.get("max_research_loops") is not None
            else configurable.max_research_loops
        )
        if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
            return "finalize_answer"
        else:
            return [
                Send(
                    "web_research",
                    {
                        "search_query": follow_up_query,
                        "id": state["number_of_ran_queries"] + int(idx),
                    },
                )
                for idx, follow_up_query in enumerate(state["follow_up_queries"])
            ]

    def finalize_answer(self, state, config):
        configurable = Configuration.from_runnable_config(config)
        reasoning_model = state.get("reasoning_model") or configurable.answer_model
        model_adapter = get_model_adapter(reasoning_model)
        current_date = get_current_date()
        formatted_prompt = answer_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            summaries="\n---\n\n".join(state["web_research_result"]),
        )
        llm = model_adapter.create_chat_model(
            model_name=reasoning_model,
            temperature=0,
            max_retries=2,
        )
        result = llm.invoke(formatted_prompt)
        unique_sources = []
        for source in state["sources_gathered"]:
            if "short_url" in source and source["short_url"] in result.content:
                result.content = result.content.replace(
                    source["short_url"], source.get("value", "")
                )
                unique_sources.append(source)
        return {
            "messages": [AIMessage(content=result.content)],
            "sources_gathered": unique_sources,
        }

# 默认实例化和编译
agent1 = Agent1(OverallState, config_schema=Configuration)
graph = agent1.compile(name="pro-search-agent") 