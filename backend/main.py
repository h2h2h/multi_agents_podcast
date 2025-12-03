import re
from typing import Any, List, Dict, Optional
import time
import struct
import queue as thread_queue
import asyncio
import threading
import os
import argparse
import signal
import json
import requests
from flask import Flask, Response, request

# AgentScope imports
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.tool import Toolkit
from agentscope.pipeline import MsgHub
from agentscope._utils import _common
import agentscope

# Local imports (Assumed to exist based on your provided code)
from utils.run_sub_lead_agent import run_sub_lead_agent, web_search
from prompts.planner_prompts import planner_sys_prompt
from prompts.sub_planner_prompts import sub_planner_sys_prompt
from prompts.role_prompts import role_sys_prompt
from prompts.director_prompts import director_sys_prompt
from prompts.screenwriter_prompts import screenwriter_sys_prompt
from utils.create_react_agent import create_local_agent
from structured.structured import (
    PodcastPlan,
    SectionContent,
    DiscussionPlan,
    PolishedScript,
)

# ======================== Configuration & Helpers ========================

TTS_API_URL = "http://xxx/v1/audio/speech"
# Voice Mapping: Map character names to specific voice IDs
VOICE_MAPPING = {
    "小明": "speech:小明:xxx:c9aaf076",
    "小红": "speech:小红:xxx:06d29d14",
    "小刚": "speech:小刚:xxx:43d2b598",
}
DEFAULT_VOICE = "001"

# Personas
PERSONAS = [
    {"name": "小明", "description": "一位资深科技专栏作家与数字化战略顾问"},
    {"name": "小红", "description": "一位独立设计师与AI艺术实践者"},
    {"name": "小刚", "description": "一位知识产权领域律师"},
]

# Server shutdown configuration (used for programmatic shutdown)
# SHUTDOWN_TOKEN = os.environ.get("SHUTDOWN_TOKEN", "localdevsecret")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8721"))
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")


def _generate_wav_header(
    data_size: int, num_channels: int = 1, sample_rate: int = 24000, bit_depth: int = 16
) -> bytes:
    header = b"RIFF"
    chunk_size = 36 + data_size
    header += struct.pack("<I", chunk_size)
    header += b"WAVE"

    header += b"fmt "
    header += struct.pack("<I", 16)
    header += struct.pack("<H", 1)
    header += struct.pack("<H", num_channels)
    header += struct.pack("<I", sample_rate)
    byte_rate = sample_rate * num_channels * bit_depth // 8
    header += struct.pack("<I", byte_rate)
    block_align = num_channels * bit_depth // 8
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bit_depth)

    header += b"data"
    header += struct.pack("<I", data_size)
    return header


def format_outline_for_prompt(outline_data: dict) -> str:
    text = f"### 主题: {outline_data['topic']}\n"
    text += f"### 摘要: {outline_data['summary']}\n\n"
    for section in outline_data["sections"]:
        text += f"### {section['section_title']}\n"
        for point in section["points"]:
            text += f"- **{point['title']}**: {point['elaboration']}\n"
    return text


def extract_json_from_text(text):
    try:
        pattern = r"\{.*\}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except Exception as e:
        print(f"JSON extraction error: {e}")
        return None


def simple_web_search(query: str, count: int = 3) -> List[str]:
    # Mocking or using the API provided in original code
    url = "https://api.bochaai.com/v1/web-search"
    payload = json.dumps({"query": query, "summary": True, "count": count})
    headers = {
        "Authorization": "sk-347ef8e6f67d4891b0f6bae3776b764f",
        "Content-Type": "application/json",
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response_data = response.json()
        return [item["summary"] for item in response_data["data"]["webPages"]["value"]]
    except Exception as e:
        print(f"[Web Search Error] {e}")
        return []


def text_to_speech(text: str, voice_id: str) -> bytes | None:
    payload = {"input": text, "voice": voice_id, "response_format": "wav", "speed": 1.0}
    try:
        response = requests.post(TTS_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"  [TTS Client] Error calling TTS API: {e}")
        return None


def parse_dialogue_line(line: str) -> tuple[str, str] | None:
    match = re.match(r"^\s*(?P<character>.+?)\s*[：:]\s*(?P<text>.+)\s*$", line)
    if match:
        return match.group("character").strip(), match.group("text").strip()
    return None


# ======================== Core Logic: Planning Phase ========================

async def generate_outline_and_knowledge(topic: str) -> tuple[Dict, List[Dict]]:
    """
    Executes the planning phase:
    1. LeadAgent searches and creates a high-level plan.
    2. SubLeadAgents fill in the details for each section and gather knowledge.
    """
    start = time.time()
    print(f"\n{'='*20}\n[Phase 1] Planning Podcast: {topic}\n{'='*20}")

    # 1. Create Planner
    planner = create_local_agent(
        name="LeadAgent",
        model_name="qwen",
        sys_prompt=planner_sys_prompt,
    )

    # 2. Initial Search
    initial_search_res = simple_web_search(topic, count=2)
    initial_context = "\n\n".join(initial_search_res)
    print(f"[Planner] Context gathered:\n{initial_context[:200]}...")

    # 3. Generate High-level Plan
    plan_msg = await planner(
        Msg(
            "user", f"本期播客主题为：{topic}\n相关背景资料：{initial_context}", "user"
        ),
        structured_model=PodcastPlan,
    )
    podcast_plan = plan_msg.metadata

    # Save artifacts
    with open("podcast_plan.json", "w", encoding="utf-8") as f:
        json.dump(podcast_plan, f, ensure_ascii=False, indent=4)

    # 4. Create Sub-Agents and Knowledge Containers
    # Knowledge is a list of dicts, one per task/section
    knowledge = [dict() for _ in range(len(podcast_plan["tasks"]))]
    sub_lead_agents = []

    for i in range(len(podcast_plan["tasks"])):
        toolkit = Toolkit()
        # Bind the specific knowledge dict to this tool instance
        toolkit.register_tool_function(
            web_search, preset_kwargs={"knowledge": knowledge[i]}
        )
        agent = create_local_agent(
            name=f"SubLeadAgent{i}",
            model_name="qwen",
            sys_prompt=sub_planner_sys_prompt,
            toolkit=toolkit,
            formatter=OpenAIChatFormatter(),
        )
        sub_lead_agents.append(agent)

    # 5. Parallel Execution of Sub-Tasks
    print("[Planner] Generating detailed sections...")
    tasks = []
    for i, task in enumerate(podcast_plan["tasks"]):
        coroutine = sub_lead_agents[i](
            Msg(
                "LeadAgent",
                f"section_title: {task['section_title']}\n"
                f"global_context: {task['global_context']}\n"
                f"specific_instruction: {task['specific_instruction']}",
                "assistant",
            ),
            structured_model=SectionContent,
        )
        tasks.append(coroutine)

    results = await asyncio.gather(*tasks)

    # 6. Assemble Outline
    sections = []
    for idx, res in enumerate(results):
        section_data = res.metadata
        if not section_data:
            # Fallback text extraction if structured parsing failed
            text = (
                res.content[0]["text"] if isinstance(res.content, list) else res.content
            )
            section_data = extract_json_from_text(text)
        sections.append(section_data)

    outline = {
        "topic": podcast_plan["topic"],
        "summary": podcast_plan["summary"],
        "sections": sections,
    }

    # Save artifacts
    with open("outline.json", "w", encoding="utf-8") as f:
        json.dump(outline, f, ensure_ascii=False, indent=4)
    with open("knowledge.json", "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=4)

    print("[Phase 1] Outline and Knowledge generated successfully.")
    end = time.time()
    print(f"[Time] Planning Phase took {end-start:.2f} seconds.")
    return outline, knowledge


# ======================== Core Logic: Dialogue Phase ========================

async def run_dialogue_loop(
    outline: Dict,
    knowledge: List[Dict],
    audio_queue: Optional[thread_queue.Queue] = None,
) -> None:
    """
    Executes the dialogue generation phase.
    If audio_queue is provided, it generates TTS audio, pushes to the queue,
    AND saves the complete audio file locally.
    """
    print(f"\n{'='*20}\n[Phase 2] Generating Dialogue & Audio\n{'='*20}")

    # 1. Initialize Agents
    director = create_local_agent(
        name="Director",
        model_name="qwen",
        sys_prompt=director_sys_prompt(
            topic=outline["topic"],
            formatted_outline=format_outline_for_prompt(outline),
            personas=PERSONAS,
        ),
    )

    roles = {}
    for p in PERSONAS:
        roles[p["name"]] = create_local_agent(
            name=p["name"],
            model_name="qwen",
            sys_prompt=role_sys_prompt(name=p["name"], persona=p["description"]),
        )

    screenwriter = create_local_agent(
        name="Screenwriter",
        model_name="qwen",
        sys_prompt=screenwriter_sys_prompt,
    )

    summary_agent = create_local_agent(
        name="SummaryAgent",
        model_name="qwen",
        sys_prompt="作为一名播客节目内容的总结者，你的任务是观察所有嘉宾对于当前章节的发言，并进行总结。",
    )

    role_list = list(roles.values())
    final_script_lines = []

    # [Added] Buffer to store full audio for file saving
    full_audio_buffer = bytearray()

    # 2. Iterate through sections and points
    current_section_idx = 0
    current_point_idx = 0

    # set a history dialogues list
    history_dialogues = []
    # set a history plan list for director
    history_plans = []

    # store sumaries for director
    summaries = []

    # set a short delay for better TTS playback
    # is_first_turn = True

    while current_section_idx < len(outline["sections"]):
        section = outline["sections"][current_section_idx]
        if not section:  # Skip empty sections
            current_section_idx += 1
            continue

        section_title = section["section_title"]
        point = section["points"][current_point_idx]
        point_title = point["title"]
        point_content = point["elaboration"]

        final_script_lines.append(f"{section_title}-{point_title}\n")

        print(f"\n--- Processing: {section['section_title']} -> {point['title']} ---")

        # Prepare Knowledge Context
        curr_know_dict = (
            knowledge[current_section_idx]
            if current_section_idx < len(knowledge)
            else {}
        )
        relevant_knowledge = "\n".join(
            [f"- {k}: {v}" for k, v in curr_know_dict.items()]
        )
        point_titles = "\n".join([f"- {p['title']}" for p in section["points"]])

        # format history dialogues
        formatted_history_dialogues_last_3 = "\n".join(history_dialogues)
        formatted_history_dialogues_last_1 = "\n".join(history_dialogues[-1:])

        # format history plans
        formatted_history_plans = "\n".join(history_plans)

        # Director Planning
        try:
            director_msg = await director(
                Msg(
                    "user",
                    f"""现在，我们将进入新的讨论环节。
- 章节: 
{section_title}

- 当前讨论点: 
{point_title}

- 具体内容: 
{point_content}

- 上一轮角色的具体发言:
{formatted_history_dialogues_last_3}

- 上一轮角色的发言计划:
{formatted_history_plans}

- 相关知识: 
{relevant_knowledge}

1. 回忆之前的讨论点，不要重复，而是要层层推进。
2. 请分析上一轮最后一位角色的发言，一定要衔接要自然，话题不能跳跃。如果在上一轮抛出了疑问，一定要安排角色进行回应。
3. 请分析上一轮角色的发言计划，不要重复，尽量能多样性。
现在，为这个讨论点策划一个详细的讨论计划。
""",
                    "user",
                ),
                structured_model=DiscussionPlan,
            )
            steps = director_msg.metadata.get("steps", [])

            # save director's plan
            with open(f"discussion_plan.json", "a", encoding="utf-8") as f:
                json.dump(director_msg.metadata, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[Error] Director failed: {e}")
            steps = []

        # Role Playing
        raw_dialogue = []
        for step in steps:
            speaker_name = step.get("speaker_name")
            instruction = step.get("instruction")
            if speaker_name not in roles:
                continue

            # update history plans
            history_plans.clear()
            history_plans.append(f"- {speaker_name}: {instruction}")

            speaker = roles[speaker_name]
            print(f"[Action] Director -> {speaker_name}: {instruction}")

            # Using MsgHub to share context among all participants
            async with MsgHub(
                # [director] + role_list + [summary_agent],
                role_list + [summary_agent],
                announcement=Msg(
                    name="Director",
                    content=f"""/no_think
接下来，请根据导演指令,你的人设和其他嘉宾的发言，生成你的发言。注意，需要自然地衔接上一个人的发言。不要重复发言。

- 导演指令: {instruction}
""",
                    role="assistant",
                ),
            ):
                try:
                    res = await speaker()
                    content = res.content[0]["text"].replace(
                        "<think>\n\n</think>\n\n", ""
                    )
                    raw_dialogue.append(f"{speaker_name}: {content}")
                    print(f"  -> {speaker_name}: {content}")
                except Exception as e:
                    print(f"  [Error] Speaker {speaker_name} failed: {e}")

        # Screenwriter Polishing
        print("  [Screenwriter] Polishing...")
        try:
            raw_text = "\n".join(raw_dialogue)
            polish_msg = await screenwriter(
                Msg(
                    "user",
                    f"""这是关于讨论点 '{point_title}' 的原始对话记:
{raw_text}                   
请根据你的专业能力，对其进行衔接、润色和优化。在此过程中，请
    - 结合上一轮对话内容，衔接要自然，消除重复对话。
    - 如果发现前后过于跳跃，有逻辑问题，可以补充发言人，使得对话更自然。

- 这是上一轮最后一位角色发言内容，供你衔接:
{formatted_history_dialogues_last_1}
""",
                    "user",
                ),
                structured_model=PolishedScript,
            )
            polished_data = polish_msg.metadata.get("polished_dialogue", [])
        except Exception as e:
            print(f"[Error] Screenwriter failed: {e}")
            polished_data = []

        # TTS & Output
        polished_lines = []
        for item in polished_data:
            line_str = f"{item['speaker']}: {item['content']}"
            polished_lines.append(line_str)

            # --- Audio Generation Logic ---
            if audio_queue is not None:
                # set a short delay for better TTS playback
                # if is_first_turn:
                #     await asyncio.sleep(5)
                #     is_first_turn = False
                parsed = parse_dialogue_line(line_str)
                if parsed:
                    char, text = parsed
                    vid = VOICE_MAPPING.get(char, DEFAULT_VOICE)
                    # Run blocking request in executor
                    audio_bytes = await asyncio.get_running_loop().run_in_executor(
                        None, text_to_speech, text, vid
                    )
                    if audio_bytes:
                        # Skip WAV header (44 bytes) for streaming raw PCM or concat
                        pcm = audio_bytes[44:] if len(audio_bytes) > 44 else audio_bytes

                        # 1. Put into streaming queue
                        audio_queue.put(pcm)
                        # 2. [Added] Append to full buffer
                        full_audio_buffer.extend(pcm)

        # update history dialogues, add last 3 turns to history
        history_dialogues.clear()
        history_dialogues.extend(raw_dialogue[-3:])

        final_script_lines.extend(polished_lines)

        # Move to next point
        current_point_idx += 1
        if current_point_idx >= len(section["points"]):
            # End of Section: Summarize and Refresh Memory
            print(f"\n[Summary] Finishing section {section['section_title']}")
            try:
                summ_res = await summary_agent(
                    Msg(
                        "user",
                        "现在，请你根据嘉宾在本轮的发言，用一句话总结本轮讨论的内容。",
                        "user",
                    )
                )
                summary_text = summ_res.content[0]["text"]
                # debug check roles,director and summary agent memory
                for ag in role_list + [director, summary_agent, screenwriter]:
                    with open(f"{ag.name}_memory.json", "w", encoding="utf-8") as f:
                        f.write(f"memory size: {await ag.memory.size()}")
                        for msg in await ag.memory.get_memory():
                            f.write(
                                f"{msg.name}: {json.dumps(msg.content, indent=4, ensure_ascii=False)}\n"
                            )

                # update summary dict
                summaries.append(Msg(
                    'system',
                    f"{section_title}-{point_title} 讨论内容总结为：{summary_text}",
                    'system'    
                ))
            except:
                summary_text = "本节讨论结束。"

            # async def trim_memory_to_last_turn(agent: ReActAgent, participants: int):
            #     full_memory = await agent.memory.get_memory()
            #     if len(full_memory) > participants:
            #         memory = full_memory[-participants:]
            #     else:
            #         memory = full_memory
            #     await agent.memory.clear()
            #     for msg in memory:
            #         await agent.memory.add(msg)

            # participants = len(step)
            # for role in role_list:
            #     await trim_memory_to_last_turn(role, (participants + 1) * 2)
            # await trim_memory_to_last_turn(director, 4)
            # # await trim_memory_to_last_turn(summary_agent,  (participants + 1) * 2 + 2)
            # await summary_agent.memory.clear()
            # all_agents = role_list + [director, summary_agent]
            # for ag in all_agents:
            #     await ag.memory.add(
            #         Msg(
            #             "system",
            #             f"上一章节标题为: {section_title}\n\n内容总结为：{summary_text}",
            #             "system",
            #         )
            #     )
            # await trim_memory_to_last_turn(screenwriter, 4)

            # ----- clear memory -----
            for ag in role_list + [summary_agent, screenwriter]:
                await ag.memory.clear()
                # add summary to memory
                await ag.memory.add(
                    Msg(
                        "system",
                        f"上一章节: {section_title}\n\n播客内容总结为：{summary_text}",
                        "system",
                    )
                )
            await director.memory.clear()
            for msg in summaries:
                await director.memory.add(msg)

            print("-----[Memory]-----")
            for role in role_list:
                print(f"{role.name} memory size: {await role.memory.size()}")
            print(f"Director memory size: {await director.memory.size()}")
            print(f"SummaryAgent memory size: {await summary_agent.memory.size()}")
            print(f"Screenwriter memory size: {await screenwriter.memory.size()}")

            current_section_idx += 1
            current_point_idx = 0

    # Save final script
    with open("podcast_script.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(final_script_lines))
    print("[Phase 2] Script generation complete.")

    # [Added] Save full audio file logic
    if audio_queue is not None and len(full_audio_buffer) > 0:
        print("[System] Saving full audio to podcast_full.wav...")
        try:
            # Assumes _generate_wav_header is defined in global scope or helpers
            header = _generate_wav_header(len(full_audio_buffer), sample_rate=24000)
            with open("podcast_full.wav", "wb") as f:
                f.write(header)
                f.write(full_audio_buffer)
            print("[System] Full audio saved successfully.")
        except Exception as e:
            print(f"[Error] Failed to save full audio file: {e}")


# ======================== Unified Pipeline Controllers ========================


async def run_text_mode(topic: str):
    """Mode 1: Generate outline, knowledge, and script (text only)."""
    # outline, knowledge = await generate_outline_and_knowledge(topic)

    # debug
    with open("outline_test.json", "r", encoding="utf-8") as f:
        outline = json.load(f)
    with open("knowledge.json", "r", encoding="utf-8") as f:
        knowledge = json.load(f)

    await run_dialogue_loop(outline, knowledge, audio_queue=None)


async def run_stream_mode(topic: str, audio_queue: thread_queue.Queue):
    """Mode 2: Generate everything and stream audio to queue."""
    try:
        # Step 1: Generate Plan (this might take time, client waits)
        # outline, knowledge = await generate_outline_and_knowledge(topic)

        # debug
        with open("outline_test.json", "r", encoding="utf-8") as f:
            outline = json.load(f)
        with open("knowledge.json", "r", encoding="utf-8") as f:
            knowledge = json.load(f)

        # Step 2: Generate Dialogue & Stream Audio
        await run_dialogue_loop(outline, knowledge, audio_queue=audio_queue)
    except Exception as e:
        print(f"[Stream Error] Pipeline failed: {e}")
    finally:
        # Sentinel to indicate stream end
        audio_queue.put(None)


# ======================== Flask & Entry Points ========================

app = Flask(__name__)


def generate_audio_stream_generator(topic: str):
    """Background thread runner for streaming."""
    q = thread_queue.Queue()

    def background_runner():
        # Create a new event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_stream_mode(topic, q))
        loop.close()

    t = threading.Thread(target=background_runner, daemon=True)
    t.start()

    # Yield audio chunks as they arrive
    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield chunk


@app.route("/stream")
def stream_audio():
    topic = request.args.get("topic", "AI Tech Trends")
    # Using L16 (PCM) mimetype. Adjust to audio/wav if sending full wav headers per chunk (not recommended for continuous stream)
    return Response(
        generate_audio_stream_generator(topic),
        mimetype="audio/L16; rate=24000; channels=1",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Podcast Generator")
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        default="Introducing Agentscope",
        help="Podcast Topic",
    )
    parser.add_argument(
        "--serve", action="store_true", help="Run in Server Mode (Streaming Audio)"
    )

    args = parser.parse_args()

    agentscope.init(studio_url="http://localhost:3000")

    if args.serve:
        print(f"Starting Streaming Server for topic: '{args.topic}'")
        print(f"Endpoint: http://0.0.0.0:{SERVER_PORT}/stream?topic=YOUR_TOPIC")
        app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True)
    else:
        print(f"Starting Text-Only Generation for topic: '{args.topic}'")
        start_t = time.time()
        asyncio.run(run_text_mode(args.topic))
        print(f"Total time elapsed: {time.time() - start_t:.2f}s")
