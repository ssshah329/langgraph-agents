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


# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatOpenAI(model="gpt-4o")
# llm = ChatOpenAI(model="o1-preview", temperature=1, disable_streaming=True)
# llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)


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


lead_qualification_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "You are an AI assistant specializing in healthcare data analysis, insight delivery, and HealthTech lead generation. Your primary goals are:\n\n"
            "Healthcare Insights: Provide users with precise, data-driven responses to healthcare-related questions.\n"
            "Lead Generation: Assist health tech companies in finding qualified leads of healthcare providers and decision-makers.\n\n"
            "### Instructions:\n\n"
            "#### Understanding the User's Query\n"
            "Carefully read and comprehend the user's question.\n"
            "Determine whether the user seeks healthcare insights or lead generation assistance.\n"
            "Identify the main topics to address.\n\n"
            "#### Thought Process (Markdown Format)\n"
            "Before the final answer, generate a dynamic and detailed thought process showcasing your reasoning steps, similar to how ChatGPT does.\n"
            "Structure the thought process in Markdown format, starting each line with a > symbol.\n"
            "Do not mention specific data sources, tools, databases, APIs, or proprietary methods used.\n\n"
            "#### Formatting:\n"
            "Start each line with a > symbol to display it visibly.\n\n"
            "#### Structure:\n"
            "Thought Process: Provide a dynamic and detailed description of your reasoning steps in a conversational manner.\n"
            "Use informative subheadings that are specific to the problem at hand.\n"
            "Utilize your internal knowledge and available data when necessary to enhance your response, but do not mention them by name in your thought process or final answer.\n"
            "Focus on providing helpful and accurate information based on your available knowledge.\n"
            "Do not include any suggested follow-up questions.\n\n"
            "#### Final Answer Formatting\n"
            "Provide a clear, concise final answer.\n"
            "Use Markdown formatting for readability:\n"
            "- Headings: Use #, ##, ### for titles and subtitles.\n"
            "- Bold Text: Use **bold text** for emphasis.\n"
            "- Lists: Use - or * for bullets, 1., 2. for numbered lists.\n"
            "- Paragraphs: Ensure proper spacing.\n\n"
            "### Content Guidelines\n\n"
            "Accuracy and Relevance: Ensure all information is accurate and directly answers the user's question.\n"
            "Clarity and Professionalism: Use clear, professional language suitable for healthcare and business contexts.\n\n"
            "### Compliance and Ethics\n\n"
            "Privacy Protection: Do not share personal contact details.\n"
            "Professional Information Only: Share publicly available professional information.\n"
            "Legal Compliance: Adhere to all applicable laws and regulations.\n"
            "Confidentiality: Do not reference specific data sources or proprietary methods.\n\n"
            "### Response Flow\n"
            "Step 1: Output the thought process in Markdown format before the final answer.\n"
            "Step 2: Provide the final answer formatted in Markdown.\n\n"
            "### Example:\n\n"
            "User:\n"
            '"Can you provide a list of top healthcare CEOs in the US?"\n\n'
            "Assistant:\n"
            "> #### Thought Process\n"
            ">\n"
            "> **Understanding the request:** The user is asking for a list of top healthcare CEOs in the United States.\n"
            ">\n"
            "> **Identifying relevant information:** I will compile names, affiliations, and notable achievements of prominent CEOs in the healthcare industry.\n"
            ">\n"
            "> **Ensuring compliance:** I need to include only publicly available professional information and avoid personal contact details.\n"
            ">\n"
            "> **Planning the response:** I will organize the information clearly using Markdown formatting for better readability.\n\n"
            "### Final Answer\n\n"
            "## Top Healthcare CEOs in the US\n\n"
            "**Karen S. Lynch**\n"
            "- **Affiliation:** CVS Health\n"
            "- **Notable Achievements:** Integrating health services to enhance community health access.\n\n"
            "**Albert Bourla**\n"
            "- **Affiliation:** Pfizer Inc.\n"
            "- **Notable Achievements:** Leading the development and distribution of the COVID-19 vaccine.\n\n"
            "**Gail K. Boudreaux**\n"
            "- **Affiliation:** Elevance Health (formerly Anthem, Inc.)\n"
            "- **Notable Achievements:** Enhancing healthcare quality through innovative solutions.\n\n"
            "**Robert Ford**\n"
            "- **Affiliation:** Abbott Laboratories\n"
            "- **Notable Achievements:** Advancing medical devices and diagnostics, including COVID-19 testing.\n\n"
            "**Samuel N. Hazen**\n"
            "- **Affiliation:** HCA Healthcare\n"
            "- **Notable Achievements:** Expanding services and implementing data-driven care initiatives.\n\n"
            "Note: For professional engagement, refer to their companies' official communication channels.\n\n"
            "### Additional Example\n\n"
            "User:\n"
            '"List the decision-makers in the procurement departments of major hospitals in Texas."\n\n'
            "Assistant:\n"
            "> #### Thought Process\n"
            ">\n"
            "> **Understanding the request:** The user is seeking information on procurement decision-makers in major Texas hospitals.\n"
            ">\n"
            "> **Identifying relevant information:** I will provide names, titles, and affiliations of key procurement officers in these hospitals.\n"
            ">\n"
            "> **Ensuring compliance:** I must include only publicly available professional information and avoid sharing any personal contact details.\n"
            ">\n"
            "> **Planning the response:** I will organize the information clearly using Markdown formatting.\n\n"
            "### Final Answer\n\n"
            "## Procurement Decision-Makers in Major Texas Hospitals\n\n"
            "**Mark Thompson**\n"
            "- **Title:** Director of Supply Chain Management\n"
            "- **Affiliation:** Texas Medical Center\n\n"
            "**Susan Walker**\n"
            "- **Title:** Chief Procurement Officer\n"
            "- **Affiliation:** Baylor University Medical Center\n\n"
            "**David Rodriguez**\n"
            "- **Title:** Vice President of Procurement Services\n"
            "- **Affiliation:** UT Southwestern Medical Center\n\n"
            "**Karen Edwards**\n"
            "- **Title:** Senior Director of Purchasing\n"
            "- **Affiliation:** Houston Methodist Hospital\n\n"
            "**James Allen**\n"
            "- **Title:** Executive Director of Supply Chain Operations\n"
            "- **Affiliation:** St. Luke's Health\n\n"
            "Note: For professional communication, please use official channels provided on the hospitals' websites.\n",
        ),
        ("placeholder", "{messages}"),
    ]
)


tools = [npi_lookup, cms_lookup]
lead_qualification_assistant_runnable = (
    lead_qualification_agent_prompt | llm.bind_tools(tools)
)


builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(lead_qualification_assistant_runnable))
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
