from typing import Literal
from agentscope.agent import ReActAgent, AgentBase
from agentscope.formatter import (
    DashScopeChatFormatter,
    OpenAIChatFormatter,
    FormatterBase,
    OpenAIMultiAgentFormatter,
    DeepSeekChatFormatter,
    DashScopeMultiAgentFormatter
)
from agentscope.memory import InMemoryMemory,MemoryBase
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel, OpenAIChatModel, ChatModelBase
from agentscope.tool import Toolkit, execute_python_code
from agentscope.plan import PlanNotebook

# you should set the base url of your local agent model here
VLLM_BASEURL = "http://xxx/v1"

def create_local_agent(
    name: str,
    model_name: Literal["qwen", "hunyuan"],
    sys_prompt: str,
    toolkit: Toolkit = None,
    plan_notebook: PlanNotebook = None,
    formatter: FormatterBase = OpenAIMultiAgentFormatter(),
) -> ReActAgent:
    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=OpenAIChatModel(
            model_name=model_name,
            api_key="dummy",
            client_args={"base_url": VLLM_BASEURL},
            generate_kwargs={
                "parallel_tool_calls": True,
                "temperature": 1.2
            },
        ),
        formatter=formatter,
        toolkit=toolkit,
        parallel_tool_calls=True,
        max_iters=3,
        plan_notebook=plan_notebook,
    )

def create_qwen_agent(
    name: str,
    sys_prompt: str,
    formatter: FormatterBase = DashScopeMultiAgentFormatter(),
    toolkit: Toolkit = None,
    plan_notebook: PlanNotebook = None,
) -> ReActAgent:
    API_KEY = "xxx"
    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=API_KEY,
            generate_kwargs={
                "parallel_tool_calls": True,
            },
        ),
        parallel_tool_calls=True,
        formatter=formatter,
        toolkit=toolkit,
        plan_notebook=plan_notebook,
    )

def create_ds_agent(
    name: str,
    sys_prompt: str,
    formatter: FormatterBase = DeepSeekChatFormatter(),
    toolkit: Toolkit = None,
    plan_notebook: PlanNotebook = None,
) -> ReActAgent:
    API_KEY = "xxx"
    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=OpenAIChatModel(
            model_name="deepseek-chat",
            api_key=API_KEY,
            client_args={"base_url": "https://api.deepseek.com/v1"},
            generate_kwargs={
                "parallel_tool_calls": True,
            },
        ),
        parallel_tool_calls=True,
        formatter=formatter,
        toolkit=toolkit,
        plan_notebook=plan_notebook,
    )
