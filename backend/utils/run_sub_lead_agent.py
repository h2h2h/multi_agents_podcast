import json
from typing import List
from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock
from agentscope.tool import Toolkit
from agentscope.memory import InMemoryMemory

from utils.web_search import web_search
from utils.run_sub_agent import run_sub_agent
from utils.create_react_agent import create_qwen_agent, create_local_agent

from pydantic import BaseModel, Field

import uuid


class Point(BaseModel):
    title: str = Field(
        ...,
        description="The core title of the sub-point. For openings, this could be 'suspense introduction'; for main content, this could be 'key argument'.",
    )
    elaboration: str = Field(
        ...,
        description=(
            "Detailed elaboration of this point. The content must have a 'podcast feel': use conversational, vivid language. "
            "For main paragraphs, it should naturally incorporate data, case studies, or specific logical reasoning. "
            "Avoid abstract concepts - present stories or analyses that listeners can easily understand."
        ),
    )


class SectionContent(BaseModel):
    section_title: str = Field(
        ...,
        description="The title of this section, which should match the input instruction.",
    )
    points: List[Point] = Field(
        ...,
        description="Several logical points that form this section. Typically 2-4 points are sufficient to support one section.",
    )


async def run_sub_lead_agent(
    section_title: str,
    global_context: str,
    specific_instruction: str,
) -> ToolResponse:
    """
    Args:
        - section_title (str): The title of the podcast section to be written.
        - global_context (str): The overall topic, summary, and tone of the podcast.
        - specific_instruction (str): Detailed instructions on what to cover in this specific section (e.g., key arguments, required data points).
    """

    sub_lead_sys_prompt = """
You are a **Senior Podcast Columnist** and **Section Writer**.
You work within a team led by an Editor-in-Chief. Your sole responsibility is to draft the content for ONE specific section of a podcast episode based on strict instructions.

### INPUT CONTEXT
You will be provided with:
<section_title>
{section_title}
</section_title>

<global_context>
{global_context}
(This includes the overall Show Title, Summary, and Tone. Align your writing style with this.)
</global_context>

<specific_instruction>
{specific_instruction}
(This is your COMMAND. You must cover these points. If it asks for specific data you don't have, you must find it.)
</specific_instruction>

### TOOL USE GUIDELINES (CRITICAL)
You have access to the `run_sub_agent` tool.
1.  **Assess Gaps**: Before writing, ask yourself: "Do I have the specific facts/stats/examples requested in `<specific_instruction>`?"
2.  **Action**:
    *   If **MISSING INFO**: Call `run_sub_agent` immediately to research that specific point.
    *   If **SUFFICIENT INFO**: Proceed to generate the final JSON content.
3.  **Prohibition**: DO NOT hallunicate specific data (numbers, dates, report names). Search for them if unknown.

### WRITING STYLE
*   **Conversational**: You are writing for audio. Use rhythm, rhetorical questions, and clear transitions.
*   **Deep**: Don't just list facts. Explain *why* it matters.
*   **Structure**: Break your section into logical `points`.

### OUTPUT FORMAT
You must output a **JSON object** matching the `SectionContent` schema.
Structure:
{
  "section_title": "...",
  "points": [
    { "title": "...", "elaboration": "..." },
    ...
  ]
}

### THINKING PROCESS
Before generating the tool call or the final JSON, analyze the request inside <thinking> tags:
1.  **Understand Intent**: What is the goal of this section (Intro, Argument, or Conclusion)?
2.  **Check Data**: Look at the `specific_instruction`. Do I need to search for external information to fulfill it?
    *   *Decision*: [SEARCH NEEDED] or [READY TO WRITE]
3.  **Drafting Strategy**: If ready, how will I structure the points to be engaging?
"""

    """
    这里生成唯一id，作为 Sub-Lead 的名称
    """
    sub_lead_name = f"SubLead_{str(uuid.uuid4())[:3]}"

    sub_lead_toolkit = Toolkit()

    sub_lead_toolkit.register_tool_function(
        run_sub_agent, preset_kwargs={"requesting_agent_name": sub_lead_name}
    )

    sub_lead_agent = create_local_agent(
        name=sub_lead_name,
        model_name="qwen", 
        sys_prompt=sub_lead_sys_prompt,
        toolkit=sub_lead_toolkit,
    )

    msg = Msg(
        name="LeadAgent",
        role="assistant",  
        content=f"""
Please write the following section:
section_title: {section_title}
global_context: {global_context}
specific_instruction: {specific_instruction}
""",
    )


    sub_lead_response = await sub_lead_agent(msg, structured_model=SectionContent)

    with open("sub_lead_debug.txt", "w", encoding="utf-8") as f:
        for msg in await sub_lead_agent.memory.get_memory():
            f.write(
                f"{msg.name}: {json.dumps(msg.content, indent=4, ensure_ascii=False)}\n",
            )
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(
                    sub_lead_response.metadata, indent=4, ensure_ascii=False
                ),
                # text=sub_lead_response.content,
            ),
        ],
    )
