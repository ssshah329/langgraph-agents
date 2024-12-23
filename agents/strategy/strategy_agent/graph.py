from datetime import datetime
from typing import Callable

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import ToolMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from strategy_agent.prompts import (
    SYSTEM_PROMPT,
    ANALYTICS_PROMPT,
    LEAD_QUALIFICATION_PROMPT,
    PROSPECTING_PROMPT,
    STRATEGY_PLANNER_PROMPT,
)
from strategy_agent.tools import cms_lookup, npi_lookup
from strategy_agent.utils import (
    create_tool_node_with_fallback,
    create_prompt,
    pop_dialog_state,
)
from strategy_agent.state import State

# LLM Setup
# llm = ChatAnthropic(model="claude-3-sonnet-20240229")
llm = ChatOpenAI(model="gpt-4o")


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


# tools
tools = [cms_lookup, npi_lookup]

# Prompts for Specialized Assistants

# Analytics assistant
anlaytics_prompt = create_prompt(ANALYTICS_PROMPT)

# Prospecting Assistant
prospecting_prompt = create_prompt(PROSPECTING_PROMPT)

# Lead qualification Assistant
lead_qualification_prompt = create_prompt(LEAD_QUALIFICATION_PROMPT)

# Strategy Assistant
strategy_prompt = create_prompt(STRATEGY_PLANNER_PROMPT)


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


primary_assistant_prompt = create_prompt(SYSTEM_PROMPT)

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
    create_tool_node_with_fallback(tools),
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
    create_tool_node_with_fallback(tools),
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
    create_tool_node_with_fallback(tools),
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
graph.name = "Strategy Agent"
