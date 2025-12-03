import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Union
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

# Configuration
API_URL = "https://api.bochaai.com/v1/web-search"
API_KEY = "xxx"

async def _execute_search_request(
    session: aiohttp.ClientSession, query: str, count: int, summary: bool = True
) -> Dict[str, Any]:
    """
    Internal helper function to execute the HTTP request for a single search query.

    Parameters:
    - session: The aiohttp client session.
    - query (str): The search query string.
    - count (int): The number of search results to return.
    - summary (bool, optional): Whether to return web page summaries. Defaults to True.

    Returns:
    - dict: A dictionary containing the query, success status, and results or error.
    """
    payload = {
        "query": query,
        "summary": summary,
        "count": count,
    }
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json",
    }

    try:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            response_data = await response.json()
            if response.status == 200 and "data" in response_data:
                relevant_info = [
                    value["summary"]
                    for value in response_data["data"]["webPages"]["value"]
                ]
                return {"query": query, "success": True, "results": relevant_info}
            else:
                return {
                    "query": query,
                    "success": False,
                    "error": response_data.get("error", "Unknown error"),
                    "results": [],
                }
    except Exception as e:
        return {"query": query, "success": False, "error": str(e), "results": []}


def _format_search_results_to_prompt(results_data: List[Dict[str, Any]]) -> str:
    """
    Helper function to format search results into a structured string for the LLM.
    """
    prompt = ""
    for item in results_data:
        prompt += f"[Search Intent]: {item['query']}\n"
        if item['success'] and item['results']:
            for idx, res in enumerate(item['results'], 1):
                prompt += f"  [{idx}] {res}\n"
        elif not item['success']:
            prompt += f"  [ERROR] {item.get('error', '')}\n"
        else:
            prompt += "  [No Results]\n"
        prompt += "\n"
    return prompt.strip()


async def web_search(
    query: str, count: int = 3, summary: bool = True, knowledge: Dict[str,str] = None
) -> ToolResponse:
    """
    Execute a single web search query.

    Parameters:
    - query (str): The search query string.
    - count (int, optional): The number of search results to return. Defaults to 5.
    - summary (bool, optional): Whether to return web page summaries. Defaults to True.

    Returns:
    - ToolResponse: The search results wrapped in an AgentScope ToolResponse.
    """
    async with aiohttp.ClientSession() as session:
        result = await _execute_search_request(session, query, count, summary)
    
    # Format the single result
    formatted_prompt = _format_search_results_to_prompt([result])

    knowledge[query] = formatted_prompt

    return ToolResponse(
        content=[
            TextBlock(type="text", text=formatted_prompt),
        ],
    )


async def parallel_web_search(
    queries: List[Dict[str, Any]], summary: bool = True
) -> ToolResponse:
    """
    Execute multiple web search queries in parallel and return formatted results.

    Parameters:
    - queries (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:
        - query (str): The search query string.
        - count (int): The number of search results to return.
    - summary (bool, optional): Whether to return web page summaries. Defaults to True.

    Returns:
    - ToolResponse: The combined search results wrapped in an AgentScope ToolResponse.

    Example:
        queries = [
            {"query": "Python Async Programming", "count": 5},
            {"query": "Machine Learning Basics", "count": 3}
        ]
        result = await parallel_web_search(queries)
    """
    if not queries:
        return ToolResponse(
            content=[
                TextBlock(type="text", text="[]"),
            ],
        )

    # Create aiohttp session and execute all searches in parallel
    async with aiohttp.ClientSession() as session:
        tasks = [
            _execute_search_request(
                session, q.get("query", ""), q.get("count", 5), summary
            )
            for q in queries
        ]
        results = await asyncio.gather(*tasks)

    # Format the results
    formatted_prompt = _format_search_results_to_prompt(results)

    return ToolResponse(
        content=[
            TextBlock(type="text", text=formatted_prompt),
        ],
    )