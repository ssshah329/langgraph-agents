from typing import Annotated, Literal
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from typing import Optional


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
