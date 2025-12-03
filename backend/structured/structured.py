from typing import Dict, List
from pydantic import BaseModel, Field

class SectionTask(BaseModel):
    """
    Defines the task for a single podcast section writer.
    """
    section_title: str = Field(
        ..., 
        description="The short, descriptive title of this specific section."
    )
    section_type: str = Field(
        ...,
        description="The structural role: 'Opening', 'Body', or 'Closing'."
    )
    global_context: str = Field(
        ..., 
        description=(
            "The unifying context string to be passed to ALL sub-agents. "
            "Must include: 1. Final Podcast Title, 2. A 150-word Summary of the whole episode (the 'North Star'), 3. The Tone/Style. "
            "NOTE: This string must be IDENTICAL for every object in the list."
        )
    )
    specific_instruction: str = Field(
        ..., 
        description=(
            "Actionable instructions for the Sub-Agent. "
            "If the Initial Research is sufficient, point them to key facts. "
            "If the Initial Research is thin on this specific angle, explicitly instruct the Sub-Agent to USE THEIR TOOLS to find new information (e.g., 'Search for latest 2024 statistics on X', 'Find a counter-argument to Y'). "
            "For Opening/Closing, specify the required ceremonial elements."
        )
    )

class PodcastPlan(BaseModel):
    """
    The strict output structure containing the list of all section tasks.
    """
    topic: str = Field(
        ..., 
        description="The overall topic of the podcast." 
    )
    summary: str = Field(
        ..., 
        description="The overall summary of the podcast."
    )
    tasks: List[SectionTask] = Field(
        ..., 
        description="A strictly ordered list of tasks starting with an Opening, followed by Body chapters, and ending with a Closing."
    )


class Point(BaseModel):
    title: str = Field(
        ..., 
        description="The core title of the sub-point. For openings, this could be 'suspense introduction'; for main content, this could be 'key argument'."
    )
    elaboration: str = Field(
        ..., 
        description=(
            "Detailed elaboration of this point. The content must have a 'podcast feel': use conversational, vivid language. "
            "For main paragraphs, it should naturally incorporate data, case studies, or specific logical reasoning. "
            "Avoid abstract concepts - present stories or analyses that listeners can easily understand."
        )
    )

class SectionContent(BaseModel):
    section_title: str = Field(
        ..., 
        description="The title of this section, which should match the input instruction."
    )
    points: List[Point] = Field(
        ..., 
        description="Several logical points that form this section. Typically 2-4 points are sufficient to support one section."
    )

class PlanStep(BaseModel):
    speaker_name: str = Field(
        description="下一步应该发言的角色名称 (例如: '小明', '小红', '小刚')"
    )
    instruction: str = Field(
        description="给这位角色的具体、个性化的指令或问题，引导他/她发言。指令应自然，并能激发角色间的互动。"
    )

class DiscussionPlan(BaseModel):
    thought: str = Field(
        description="作为导演，你在策划这个讨论点时的思考过程。比如，你为什么这样安排顺序，预期会出现什么样的互动或冲突。"
    )
    steps: List[PlanStep] = Field(
        description="为当前讨论点策划的一系列有序的发言步骤。"
    )

class Speech(BaseModel):
    speaker: str = Field(
        ..., 
        description="发言者的名称, 例如: '小明', '小红', '小刚'。"
    )
    content: str = Field(
        ..., 
        description="润色后的发言内容。"
    )

class PolishedScript(BaseModel):
    thought: str = Field(
        description="你作为编剧，在进行本次润色时的主要思路。例如，你觉得哪些地方衔接不好，你如何打散长句，以及你加入了哪些梗或金句。"
    )
    polished_dialogue: List[Speech] = Field(
        description="经过你润色后的、更加口语化、流畅自然的最终对话脚本。"
    )

