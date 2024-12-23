from typing import Annotated, Callable, Literal
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from strategy_agent.utils import update_dialog_stack


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
