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
         # 直接返回本地生成的内容，不联网
        fake_result = f"本地模拟搜索结果：关于“{state['search_query']}”的简要说明。"
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [fake_result],
    }
        # configurable = Configuration.from_runnable_config(config)
        # formatted_prompt = web_searcher_instructions.format(
        #     current_date=get_current_date(),
        #     research_topic=state["search_query"],
        # )
        # response = genai_client.models.generate_content(
        #     model="gemini-2.0-flash",
        #     contents=formatted_prompt,
        #     config={
        #         "tools": [{"google_search": {}}],
        #         "temperature": 0,
        #     },
        # )
        # resolved_urls = resolve_urls(
        #     response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
        # )
        # citations = get_citations(response, resolved_urls)
        # modified_text = insert_citation_markers(response.text, citations)
        # sources_gathered = [item for citation in citations for item in citation["segments"]]
        # return {
        #     "sources_gathered": sources_gathered,
        #     "search_query": [state["search_query"]],
        #     "web_research_result": [modified_text],
        # }

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
            if source["short_url"] in result.content:
                result.content = result.content.replace(
                    source["short_url"], source["value"]
                )
                unique_sources.append(source)
        return {
            "messages": [AIMessage(content=result.content)],
            "sources_gathered": unique_sources,
        }

# 默认实例化和编译
agent1 = Agent1(OverallState, config_schema=Configuration)
graph = agent1.compile(name="pro-search-agent") 