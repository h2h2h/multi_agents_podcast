from typing import List
import uuid
from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock
from agentscope.tool import Toolkit
from agentscope.memory import InMemoryMemory

from utils.web_search import web_search
from utils.create_react_agent import create_qwen_agent, create_local_agent

async def run_sub_agent(
    main_query: str, background_info: str, requesting_agent_name: str
) -> ToolResponse:
    """
    Args:
        - main_query (str): The main query for the sub-agent to search and gather information on.
        - background_info (str): Any additional context or background information to assist the sub-agent.
    """

    sub_agent_sys_prompt = """
<system_instructions>
<role>
You are an expert **Sub-Agent Researcher**. Your sole purpose is to gather high-fidelity information for a Lead Agent using the `web_search` tool. You act as an intelligent filter, converting broad requests into precise, verified data.
</role>

<inputs>
You will receive two inputs:
1. <main_query>: The specific topic or question you need to investigate.
2. <background_info>: Context to help you understand the nuance of the request.
</inputs>

<workflow>
1. **Analyze and Decompose (CRITICAL)**
   - Before calling any tools, analyze the `<main_query>` inside `<thinking>` tags.
   - Determine if the query contains multiple distinct entities, sub-questions, or logical steps.
   - **Rule:** If the query is complex, you MUST break it down. Do not combine unrelated search targets into a single query string.

2. **Execute Strategic Search**
   - Issue `web_search` calls.
   - **Precision Principle:** Each tool call must address **one** specific information gap.
   - *Example:* Instead of searching "Apple vs Microsoft 2024 revenue", issue one search for "Apple 2024 revenue report" and a second search for "Microsoft 2024 revenue report".

3. **Evaluate and Refine**
   - Critically assess the tool outputs.
   - If results are vague, outdated, or SEO-spam: **Refine your search terms** and search again. Do not settle for low-quality data.

4. **Synthesize Output**
   - Compile findings into a concise, dense summary optimized for another AI (the Lead Agent) to read.
   - **Strip away** all introductory fluff, conversational fillers, and marketing noise.
   - Focus purely on: **Atomic Facts, Statistics, Dates, and Verified Claims**.
   - Include source names/URLs where available.
</workflow>

<constraints>
- **PRECISION > BREADTH**: Better to make 3 specific searches than 1 broad search.
- **NO FLUFF**: Do not start with "Here is what I found." Just list the data.
- **NO HALLUCINATION**: If the tool returns no results after multiple attempts, state "No data found" rather than inventing facts.
</constraints>

<thinking_process>
Always start your response with a <thinking> block:
1. Identify the core intent of the <main_query>.
2. List the specific, separate search queries needed to answer it comprehensively.
3. specificy the order of execution.
</thinking_process>
</system_instructions>
"""

    sub_agent_name = f"SubLead_{str(uuid.uuid4())[:3]}"
    print(f"Creating Sub-Agent: {sub_agent_name}")
    print(f"Serving Requesting Agent: {requesting_agent_name}")

    sub_agent_toolkit = Toolkit()
    sub_agent_toolkit.register_tool_function(web_search)

    # 创建 Researcher Agent
    sub_agent = create_local_agent(
        name=sub_agent_name,
        model_name="qwen",
        sys_prompt=sub_agent_sys_prompt,
        toolkit=sub_agent_toolkit,
    )


    msg = Msg(
        name=requesting_agent_name,
        role="assistant",  
        content=f"""
I am agent '{requesting_agent_name}'. 
I am writing a section for a podcast and I need specific evidence.

Target Query: {main_query}
Context/Background: {background_info}
""",
    )

    sub_agent_response = await sub_agent(msg)

    # Debugging
    with open("sub_agent_debug.txt", "w", encoding="utf-8") as f:
        f.write(f"[{sub_agent_name}] serving [{requesting_agent_name}]:\n")
        f.write(f"Query: {main_query}\n")
        f.write(f"Response: {sub_agent_response.content}\n{'-'*30}\n")

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=sub_agent_response.content,
            ),
        ],
    )
