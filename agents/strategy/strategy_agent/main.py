import os
import getpass
from datetime import datetime
from typing import Annotated, Callable, Literal, Optional

import requests
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage
from langgraph.graph.message import AnyMessage, add_messages

# LLM Setup
# llm = ChatAnthropic(model="claude-3-sonnet-20240229")
llm = ChatOpenAI(model="gpt-4o")

# Environment Configuration
DIFY_BASE_URL = os.environ.get("DIFY_BASE_URL")
CMS_KNOWLEDGE_BASE_ID = os.environ.get("CMS_KNOWLEDGE_BASE_ID")
NPI_KNOWLEDGE_BASE_ID = os.environ.get("NPI_KNOWLEDGE_BASE_ID")
DIFY_API_KEY = os.environ.get("DIFY_API_KEY")


# Tool Definitions
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


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


# State and Assistant Configuration
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "analytics_assistant",
                "prospecting_assistant",
                "lead_qualification_assistant",
                "strategy_planner_assistant",
            ]
        ],
        update_dialog_stack,
    ]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

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


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the database for more information.",
            },
        }


def update_dialog_state(left: list[str], right: Optional[str]) -> list[str]:
    if right == "pop":
        return left[:-1]
    return left + [right]


# Prompts for Specialized Assistants
def create_prompt(template: str):
    return ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now())


# Analytics assistant
anlaytics_prompt = create_prompt(
    "You are a specialized **Analytics Assistant** for healthcare data analysis and insights delivery. "
    "The main assistant delegates tasks to you when the user requires insights, trends, or analysis of healthcare data. "
    "Your role includes analyzing healthcare datasets, summarizing trends, identifying patterns, and visualizing insights.\n\n"
    "### Instructions:\n"
    "1. Analyze user queries and determine the type of analysis required (trends, benchmarks, outlier detection, etc.).\n"
    "2. Use tools to retrieve relevant healthcare data and generate summaries, insights, or visualizations.\n"
    "3. Present findings in a concise and structured format using clear headings and bullet points.\n\n"
    "### Capabilities:\n"
    "- Generate descriptive statistics for datasets.\n"
    "- Perform trend analysis over time.\n"
    "- Create visualizations (line charts, bar charts) for insights.\n"
    "- Summarize categorical data.\n\n"
    "### Escalation:\n"
    "If additional tools are required or the user's needs go beyond data analysis, escalate the task to the main assistant "
    'using "CompleteOrEscalate".\n\n'
    "**Example Escalations:**\n"
    "- The user requests strategy recommendations based on the analysis.\n"
    "- The user changes the task to lead qualification.\n\n"
    "### Current Time: {time}",
)

# Prospecting Assistant
prospecting_prompt = create_prompt(
    "You are a specialized **Prospecting Assistant** focused on identifying healthcare leads. "
    "The main assistant delegates tasks to you when the user requests healthcare provider leads or contact information. "
    "Your role is to find, extract, and organize leads based on the user's query.\n\n"
    "### Instructions:\n"
    "1. Interpret user queries to determine lead criteria (e.g., roles, organizations, locations).\n"
    "2. Search the available databases for healthcare providers or decision-makers.\n"
    "3. Return structured lead information, including:\n"
    "   - Name\n"
    "   - Title/Role\n"
    "   - Organization\n\n"
    "### Capabilities:\n"
    "- Search NPI or CMS data for healthcare providers.\n"
    "- Filter and organize leads based on relevance.\n"
    "- Escalate when leads require further qualification.\n\n"
    "### Escalation:\n"
    'If the task requires further lead qualification or strategy planning, escalate using "CompleteOrEscalate".\n\n'
    "**Example Escalations:**\n"
    "- User asks for lead qualification based on lead relevance.\n"
    "- User requests a marketing strategy for the identified leads.\n\n"
    "### Current Time: {time}",
)

# Lead qualification Assistant
lead_qualification_prompt = create_prompt(
    "You are a specialized **Lead Qualification Assistant** responsible for evaluating healthcare leads. "
    "The main assistant delegates tasks to you when the user needs to assess the quality or relevance of identified leads. "
    "Your role is to qualify leads based on specified criteria and provide a confidence score or summary.\n\n"
    "### Instructions:\n"
    "1. Analyze the provided leads to determine their fit for the user's goals (e.g., relevance, role, organization).\n"
    "2. Assess and prioritize leads based on user-provided criteria.\n"
    "3. Return a structured qualification summary, including:\n"
    "   - Lead Name\n"
    "   - Qualification Status (High, Medium, Low)\n"
    "   - Notes explaining the reasoning.\n\n"
    "### Capabilities:\n"
    "- Qualify leads based on user criteria.\n"
    "- Provide confidence scores and prioritization.\n"
    "- Escalate tasks if leads require further analysis or strategic planning.\n\n"
    "### Escalation:\n"
    'If further analysis, prospecting, or strategy planning is required, escalate using "CompleteOrEscalate".\n\n'
    "**Example Escalations:**\n"
    "- User requests outreach strategies for the qualified leads.\n"
    "- User changes the focus to trend analysis.\n\n"
    "### Current Time: {time}",
)

# Strategy Assistant
strategy_prompt = create_prompt(
    "You are a specialized **Strategy Planner Assistant** focused on generating outreach and marketing strategies. "
    "The main assistant delegates tasks to you when the user needs strategic recommendations for healthcare leads or insights.\n\n"
    "### Instructions:\n"
    "1. Use provided insights, qualified leads, and user goals to develop an actionable strategy.\n"
    "2. Break the strategy into clear steps, including:\n"
    "   - Target Audience\n"
    "   - Outreach Channels (e.g., email, LinkedIn)\n"
    "   - Key Messaging\n"
    "   - Call-to-Action\n\n"
    "3. Present the strategy in a structured format using headings, bullet points, and clear summaries.\n\n"
    "### Capabilities:\n"
    "- Create personalized outreach strategies.\n"
    "- Recommend marketing channels and messaging.\n"
    "- Escalate if additional data analysis or lead updates are needed.\n\n"
    "### Escalation:\n"
    'If additional data, lead identification, or further qualification is required, escalate using "CompleteOrEscalate".\n\n'
    "**Example Escalations:**\n"
    "- User requests a trend analysis before planning the strategy.\n"
    "- User changes the task to finding new leads.\n\n"
    "### Current Time: {time}"
)


# Runnable Definitions
analytics_runnable = anlaytics_prompt | llm.bind_tools(
    [cms_lookup, npi_lookup] + [CompleteOrEscalate]
)
prospecting_runnable = prospecting_prompt | llm.bind_tools(
    [cms_lookup, npi_lookup] + [CompleteOrEscalate]
)
lead_qualification_runnable = lead_qualification_prompt | llm.bind_tools(
    [cms_lookup, npi_lookup] + [CompleteOrEscalate]
)
strategy_runnable = strategy_prompt | llm.bind_tools(
    [cms_lookup, npi_lookup] + [CompleteOrEscalate]
)


# Primary Assistant
class ToAnalyticsAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle healthcare analytics."""

    request: str = Field(
        description="Any necessary followup questions the update analytics assistant should clarify before proceeding."
    )


class ToLeadQualification(BaseModel):
    """Transfers work to a specialized assistant to handle lead qualification."""

    request: str = Field(
        description="Any additional information or requests from the user regarding the lead qualification."
    )


class ToProspectingAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle prospecting leads."""

    request: str = Field(
        description="Any additional information or requests from the user regarding the prospecting leads."
    )


class ToStrategyAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle strategy planning."""

    request: str = Field(
        description="Any additional information or requests from the user regarding strategy planning."
    )


primary_assistant_prompt = create_prompt(
    "You are the **Primary Orchestration Assistant**, responsible for coordinating tasks across specialized agents. "
    "You handle user queries and delegate tasks to the appropriate assistant based on the user's intent. "
    "You do not perform these tasks directly; instead, you quietly invoke specialized agents without mentioning them to the user.\n\n"
    "### Specialized Agents:\n"
    "1. **Analytics Assistant**: Handles data analysis, insights, trends, and visualizations.\n"
    "2. **Prospecting Assistant**: Finds and provides healthcare leads.\n"
    "3. **Lead Qualification Assistant**: Qualifies and evaluates leads for relevance.\n"
    "4. **Strategy Planner Assistant**: Generates outreach and marketing strategies based on leads or insights.\n\n"
    "### Instructions:\n"
    "1. Analyze the user's query to determine the task type:\n"
    "   - **Analytics**: Queries about data insights, trends, or analysis.\n"
    "   - **Prospecting**: Requests to find healthcare leads or contacts.\n"
    "   - **Lead Qualification**: Tasks involving lead evaluation or prioritization.\n"
    "   - **Strategy Planning**: Requests for outreach strategies or recommendations.\n\n"
    "2. Delegate the task to the appropriate specialized assistant using the corresponding tool:\n"
    "   - `ToAnalyticsAssistant` for data analysis tasks.\n"
    "   - `ToProspectingAssistant` for lead identification.\n"
    "   - `ToLeadQualification` for lead qualification.\n"
    "   - `ToStrategyAssistant` for strategy planning.\n\n"
    "3. If a user query is unrelated to these tasks, respond with general information or escalate if needed.\n\n"
    "### Escalation:\n"
    "If a specialized assistant cannot handle the task (e.g., tool limitations or user changing focus), they will escalate the task back to you. "
    "You must re-assess the user's query and re-route it appropriately.\n\n"
    "### Guidelines:\n"
    "- **Be Persistent**: If searches or tasks return no results initially, expand your scope before giving up.\n"
    "- **Do Not Reveal Agents**: The user should not be aware of the specialized assistants. Present results as if they came from you.\n"
    "- **Accuracy**: Double-check all outputs and databases before concluding that information is unavailable.\n"
    "- **Escalate Appropriately**: If tools cannot resolve the query, gracefully escalate to avoid wasting the user's time.\n\n"
    "### Examples of Routing:\n"
    "1. **User**: 'Can you analyze trends in patient admissions for last year?'\n"
    "   **Action**: Use `ToAnalyticsAssistant`.\n\n"
    "2. **User**: 'Find procurement heads in hospitals around Texas.'\n"
    "   **Action**: Use `ToProspectingAssistant`.\n\n"
    "3. **User**: 'Which of these leads are most relevant for our sales team?'\n"
    "   **Action**: Use `ToLeadQualification`.\n\n"
    "4. **User**: 'Create a strategy for reaching out to hospital CEOs.'\n"
    "   **Action**: Use `ToStrategyAssistant`.\n\n"
    "### Current User Context:\n"
    "### Current Time: {time}",
)

assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    [
        ToAnalyticsAssistant,
        ToLeadQualification,
        ToProspectingAssistant,
        ToStrategyAssistant,
    ]
)


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n Please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# Build StateGraph
builder = StateGraph(State)


builder.add_edge(START, "primary_assistant")

builder.add_node(
    "enter_analytics_assistant",
    create_entry_node("Healthcare Analytics Assistant", "analytics_assistant"),
)
builder.add_node("analytics_assistant", Assistant(analytics_runnable))
builder.add_edge("enter_analytics_assistant", "analytics_assistant")
builder.add_node(
    "analytics_tools",
    create_tool_node_with_fallback([cms_lookup, npi_lookup]),
)


# Routing Logic
def route_analytics_assistant(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "analytics_tools"


# Edges for Analytics Assistant
builder.add_edge("analytics_tools", "analytics_assistant")
builder.add_conditional_edges(
    "analytics_assistant",
    route_analytics_assistant,
    ["analytics_tools", "leave_skill", END],
)


# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")


# Entry Node for Prospecting Assistant
builder.add_node(
    "enter_prospecting_assistant",
    create_entry_node("Prospecting Assistant", "prospecting_assistant"),
)
builder.add_node("prospecting_assistant", Assistant(prospecting_runnable))
builder.add_edge("enter_prospecting_assistant", "prospecting_assistant")
builder.add_node(
    "prospecting_tools",
    create_tool_node_with_fallback([cms_lookup, npi_lookup]),
)


# Routing Logic
def route_prospecting_assistant(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "prospecting_tools"


# Edges for Prospecting Assistant
builder.add_edge("prospecting_tools", "prospecting_assistant")
builder.add_conditional_edges(
    "prospecting_assistant",
    route_prospecting_assistant,
    ["prospecting_tools", "leave_skill", END],
)


# Entry Node for Lead Qualification Assistant
builder.add_node(
    "enter_lead_qualification",
    create_entry_node("Lead Qualification Assistant", "lead_qualification_assistant"),
)
builder.add_node("lead_qualification_assistant", Assistant(lead_qualification_runnable))
builder.add_edge("enter_lead_qualification", "lead_qualification_assistant")
builder.add_node(
    "lead_qualification_tools",
    create_tool_node_with_fallback([cms_lookup, npi_lookup]),
)


# Routing Logic
def route_lead_qualification(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "lead_qualification_tools"


# Edges for Lead Qualification Assistant
builder.add_edge("lead_qualification_tools", "lead_qualification_assistant")
builder.add_conditional_edges(
    "lead_qualification_assistant",
    route_lead_qualification,
    ["lead_qualification_tools", "leave_skill", END],
)


# Entry Node for Strategy Planner Assistant
builder.add_node(
    "enter_strategy_planner",
    create_entry_node("Strategy Planner Assistant", "strategy_planner_assistant"),
)
builder.add_node("strategy_planner_assistant", Assistant(strategy_runnable))
builder.add_edge("enter_strategy_planner", "strategy_planner_assistant")
builder.add_node(
    "strategy_tools",
    create_tool_node_with_fallback([cms_lookup, npi_lookup]),
)


# Routing Logic
def route_strategy_planner(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "strategy_tools"


# Edges for Strategy Planner Assistant
builder.add_edge("strategy_tools", "strategy_planner_assistant")
builder.add_conditional_edges(
    "strategy_planner_assistant",
    route_strategy_planner,
    ["strategy_tools", "leave_skill", END],
)


# Primary Assistant Node
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback([cms_lookup, npi_lookup])
)


# Routing Logic for Specialized Assistants
def route_primary_assistant(state: State):
    """
    Route tasks to the appropriate specialized assistant or tools based on tool calls.
    """
    route = tools_condition(state)
    if route == END:
        return END

    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToAnalyticsAssistant.__name__:
            return "enter_analytics_assistant"
        elif tool_calls[0]["name"] == ToProspectingAssistant.__name__:
            return "enter_prospecting_assistant"
        elif tool_calls[0]["name"] == ToLeadQualification.__name__:
            return "enter_lead_qualification"
        elif tool_calls[0]["name"] == ToStrategyAssistant.__name__:
            return "enter_strategy_planner"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


# Conditional Edges for Routing
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_analytics_assistant",
        "enter_prospecting_assistant",
        "enter_lead_qualification",
        "enter_strategy_planner",
        "primary_assistant_tools",
        END,
    ],
)
builder.add_edge("primary_assistant_tools", "primary_assistant")


# Compile Graph
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
)
