import os
import getpass
import requests
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
import uuid

# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatOpenAI(model="gpt-4o")
# llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)


# Environment Configuration
DIFY_BASE_URL = os.environ.get("DIFY_BASE_URL")
CMS_KNOWLEDGE_BASE_ID = os.environ.get("CMS_KNOWLEDGE_BASE_ID")
NPI_KNOWLEDGE_BASE_ID = os.environ.get("NPI_KNOWLEDGE_BASE_ID")
DIFY_API_KEY = os.environ.get("DIFY_API_KEY")


@tool
def npi_lookup(query: str) -> str:
    """
    Query the Dify knowledge base for relevant documents using the /retrieve endpoint.
    Returns the top results combined into a single string.
    """
    url = f"{DIFY_BASE_URL}/v1/datasets/{NPI_KNOWLEDGE_BASE_ID}/retrieve"
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "retrieval_model": {
            "search_method": "hybrid_search",  # choose from: keyword_search, semantic_search, full_text_search, hybrid_search
            "reranking_enable": False,  # False if reranking not needed
            "reranking_mode": None,  # null equivalent in Python is None
            "reranking_model": {
                "reranking_provider_name": "",
                "reranking_model_name": "",
            },
            "weights": 0.7,  # null equivalent in Python is None
            "top_k": 3,  # number of results to return
            "score_threshold_enabled": False,  # disable score threshold
            "score_threshold": None,  # null equivalent
        },
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    records = data.get("records", [])
    contents = []
    for record in records:
        segment = record.get("segment", {})
        content = segment.get("content", "")
        if content:
            contents.append(content.strip())

    return "\n\n".join(contents)


@tool
def cms_lookup(query: str) -> str:
    """
    Query the Dify knowledge base for relevant documents using the /retrieve endpoint.
    Returns the top results combined into a single string.
    """
    url = f"{DIFY_BASE_URL}/v1/datasets/{CMS_KNOWLEDGE_BASE_ID}/retrieve"
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "retrieval_model": {
            "search_method": "hybrid_search",  # choose from: keyword_search, semantic_search, full_text_search, hybrid_search
            "reranking_enable": False,  # False if reranking not needed
            "reranking_mode": None,  # null equivalent in Python is None
            "reranking_model": {
                "reranking_provider_name": "",
                "reranking_model_name": "",
            },
            "weights": 0.7,  # null equivalent in Python is None
            "top_k": 3,  # number of results to return
            "score_threshold_enabled": False,  # disable score threshold
            "score_threshold": None,  # null equivalent
        },
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    records = data.get("records", [])
    contents = []
    for record in records:
        segment = record.get("segment", {})
        content = segment.get("content", "")
        if content:
            contents.append(content.strip())

    return "\n\n".join(contents)


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


strategy_planner_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the **Strategy Planner Agent**, specializing in creating personalized marketing and lead generation strategies for healthcare tech companies. Your responsibilities include:\n\n"
            "**Primary Goals:**\n"
            "- Develop data-driven strategies for lead generation and outreach.\n"
            "- Provide actionable recommendations to maximize success.\n\n"
            "### Instructions:\n\n"
            "#### Understanding the Query\n"
            "- Analyze the query to identify the user’s marketing or outreach goals.\n"
            "- Tailor strategies to the healthcare tech industry.\n\n"
            "#### Strategy Development\n"
            "1. Define key goals and success metrics.\n"
            "2. Create a structured plan, including audience, channels, and messaging.\n"
            "3. Offer actionable steps for execution.\n\n"
            "#### Formatting the Response\n"
            "- Use **Markdown** with clear sections, bullet points, and bold highlights.\n\n"
            "---\n\n"
            "### Example\n\n"
            '**User Query:** "Create a strategy for reaching out to procurement heads."\n\n'
            "**Strategy Planner Agent:**\n\n"
            "> **Understanding the Query:** The user needs a targeted outreach strategy for procurement leaders.\n\n"
            "### Final Answer\n\n"
            "## Procurement Outreach Strategy\n\n"
            "**Goals:**\n"
            "- Reach procurement leaders at top hospitals.\n\n"
            "**Steps:**\n"
            "1. **Audience Targeting:** Focus on hospitals with $100M+ annual budgets.\n"
            "2. **Channels:**\n   - Email outreach.\n   - LinkedIn personalized messages.\n3. **Messaging:** Highlight cost-saving solutions and efficiency improvements.\n\n"
            "---",
        ),
        ("placeholder", "{messages}"),
    ]
)


tools = [TavilySearchResults(max_results=1), npi_lookup, cms_lookup]
strategy_planner_runnable = strategy_planner_agent_prompt | llm.bind_tools(tools)

builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(strategy_planner_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
