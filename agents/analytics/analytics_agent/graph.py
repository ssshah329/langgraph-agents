from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.prebuilt import tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from analytics_agent.prompts import SYSTEM_PROMPT
from analytics_agent.tools import cms_lookup, npi_lookup
from analytics_agent.utils import create_tool_node_with_fallback, create_prompt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START


# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatOpenAI(model="gpt-4o")
# llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)


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


analytics_agent_prompt = create_prompt(SYSTEM_PROMPT)


tools = [npi_lookup, cms_lookup]
analytics_assistant_runnable = analytics_agent_prompt | llm.bind_tools(tools)


builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(analytics_assistant_runnable))
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

graph.name = "Analytics Agent"
