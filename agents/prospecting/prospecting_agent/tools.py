import os
import requests
from langchain_core.tools import tool

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
