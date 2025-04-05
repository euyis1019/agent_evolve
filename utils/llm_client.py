"""LLM client for Anthropic models."""

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Tuple, cast
from dataclasses_json import DataClassJsonMixin
import anthropic
import openai
import uuid

from anthropic import (
    NOT_GIVEN as Anthropic_NOT_GIVEN,
)
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    InternalServerError as AnthropicInternalServerError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from anthropic._exceptions import (
    OverloadedError as AnthropicOverloadedError,  # pyright: ignore[reportPrivateImportUsage]
)
from anthropic.types import (
    TextBlock as AnthropicTextBlock,
    ThinkingBlock as AnthropicThinkingBlock,
    RedactedThinkingBlock as AnthropicRedactedThinkingBlock,
)
from anthropic.types import ToolParam as AnthropicToolParam
from anthropic.types import (
    ToolResultBlockParam as AnthropicToolResultBlockParam,
)
from anthropic.types import (
    ToolUseBlock as AnthropicToolUseBlock,
)
from anthropic.types.message_create_params import (
    ToolChoiceToolChoiceAny,
    ToolChoiceToolChoiceAuto,
    ToolChoiceToolChoiceTool,
)

from openai import (
    APIConnectionError as OpenAI_APIConnectionError,
)
from openai import (
    InternalServerError as OpenAI_InternalServerError,
)
from openai import (
    RateLimitError as OpenAI_RateLimitError,
)
from openai._types import (
    NOT_GIVEN as OpenAI_NOT_GIVEN,  # pyright: ignore[reportPrivateImportUsage]
)
import litellm
from litellm import completion
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class ToolParam(DataClassJsonMixin):
    """Internal representation of LLM tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall(DataClassJsonMixin):
    """Internal representation of LLM-generated tool call."""

    tool_call_id: str
    tool_name: str
    tool_input: Any


@dataclass
class ToolResult(DataClassJsonMixin):
    """Internal representation of LLM tool result."""

    tool_call_id: str
    tool_name: str
    tool_output: Any


@dataclass
class ToolFormattedResult(DataClassJsonMixin):
    """Internal representation of formatted LLM tool result."""

    tool_call_id: str
    tool_name: str
    tool_output: str


@dataclass
class TextPrompt(DataClassJsonMixin):
    """Internal representation of user-generated text prompt."""

    text: str


@dataclass
class TextResult(DataClassJsonMixin):
    """Internal representation of LLM-generated text result."""

    text: str


AssistantContentBlock = (
    TextResult | ToolCall | AnthropicRedactedThinkingBlock | AnthropicThinkingBlock
)
UserContentBlock = TextPrompt | ToolFormattedResult
GeneralContentBlock = UserContentBlock | AssistantContentBlock
LLMMessages = list[list[GeneralContentBlock]]


class LLMClient:
    """A client for LLM APIs for the use in agents."""

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.

        Args:
            messages: A list of messages.
            max_tokens: The maximum number of tokens to generate.
            system_prompt: A system prompt.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.

        Returns:
            A generated response.
        """
        raise NotImplementedError


def recursively_remove_invoke_tag(obj):
    """Recursively remove the </invoke> tag from a dictionary or list."""
    result_obj = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            result_obj[key] = recursively_remove_invoke_tag(value)
    elif isinstance(obj, list):
        result_obj = [recursively_remove_invoke_tag(item) for item in obj]
    elif isinstance(obj, str):
        if "</invoke>" in obj:
            result_obj = json.loads(obj.replace("</invoke>", ""))
        else:
            result_obj = obj
    else:
        result_obj = obj
    return result_obj


class AnthropicDirectClient(LLMClient):
    """Use Anthropic models via first party API."""

    def __init__(
        self,
        model_name="claude-3-7-sonnet-20250219",
        max_retries=2,
        use_caching=True,
        use_low_qos_server: bool = False,
        thinking_tokens: int = 0,
    ):
        """Initialize the Anthropic first party client."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        # Disable retries since we are handling retries ourselves.
        self.client = anthropic.Anthropic(
            api_key=api_key, max_retries=1, timeout=60 * 5, 
            base_url="https://chat.cloudapi.vip/"
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.use_caching = use_caching
        self.prompt_caching_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        self.thinking_tokens = thinking_tokens

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.

        Args:
            messages: A list of messages.
            max_tokens: The maximum number of tokens to generate.
            system_prompt: A system prompt.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.

        Returns:
            A generated response.
        """

        # Turn GeneralContentBlock into Anthropic message format
        anthropic_messages = []
        for idx, message_list in enumerate(messages):
            role = "user" if idx % 2 == 0 else "assistant"
            message_content_list = []
            for message in message_list:
                # Check string type to avoid import issues particularly with reloads.
                if str(type(message)) == str(TextPrompt):
                    message = cast(TextPrompt, message)
                    message_content = AnthropicTextBlock(
                        type="text",
                        text=message.text,
                    )
                elif str(type(message)) == str(TextResult):
                    message = cast(TextResult, message)
                    message_content = AnthropicTextBlock(
                        type="text",
                        text=message.text,
                    )
                elif str(type(message)) == str(ToolCall):
                    message = cast(ToolCall, message)
                    message_content = AnthropicToolUseBlock(
                        type="tool_use",
                        id=message.tool_call_id,
                        name=message.tool_name,
                        input=message.tool_input,
                    )
                elif str(type(message)) == str(ToolFormattedResult):
                    message = cast(ToolFormattedResult, message)
                    message_content = AnthropicToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=message.tool_call_id,
                        content=message.tool_output,
                    )
                elif str(type(message)) == str(AnthropicRedactedThinkingBlock):
                    message = cast(AnthropicRedactedThinkingBlock, message)
                    message_content = message
                elif str(type(message)) == str(AnthropicThinkingBlock):
                    message = cast(AnthropicThinkingBlock, message)
                    message_content = message
                else:
                    print(
                        f"Unknown message type: {type(message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
                    )
                    raise ValueError(
                        f"Unknown message type: {type(message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
                    )
                message_content_list.append(message_content)

            # Anthropic supports up to 4 cache breakpoints, so we put them on the last 4 messages.
            if self.use_caching and idx >= len(messages) - 4:
                if isinstance(message_content_list[-1], dict):
                    message_content_list[-1]["cache_control"] = {"type": "ephemeral"}
                else:
                    message_content_list[-1].cache_control = {"type": "ephemeral"}

            anthropic_messages.append(
                {
                    "role": role,
                    "content": message_content_list,
                }
            )

        if self.use_caching:
            extra_headers = self.prompt_caching_headers
        else:
            extra_headers = None

        # Turn tool_choice into Anthropic tool_choice format
        if tool_choice is None:
            tool_choice_param = Anthropic_NOT_GIVEN
        elif tool_choice["type"] == "any":
            tool_choice_param = ToolChoiceToolChoiceAny(type="any")
        elif tool_choice["type"] == "auto":
            tool_choice_param = ToolChoiceToolChoiceAuto(type="auto")
        elif tool_choice["type"] == "tool":
            tool_choice_param = ToolChoiceToolChoiceTool(
                type="tool", name=tool_choice["name"]
            )
        else:
            raise ValueError(f"Unknown tool_choice type: {tool_choice['type']}")

        if len(tools) == 0:
            tool_params = Anthropic_NOT_GIVEN
        else:
            tool_params = [
                AnthropicToolParam(
                    input_schema=tool.input_schema,
                    name=tool.name,
                    description=tool.description,
                )
                for tool in tools
            ]

        response = None

        if thinking_tokens is None:
            thinking_tokens = self.thinking_tokens
        if thinking_tokens and thinking_tokens > 0:
            extra_body = {
                "thinking": {"type": "enabled", "budget_tokens": thinking_tokens}
            }
            temperature = 1
            assert max_tokens >= 32_000 and thinking_tokens <= 8192, (
                f"As a heuristic, max tokens {max_tokens} must be >= 32k and thinking tokens {thinking_tokens} must be < 8k"
            )
        else:
            extra_body = None

        for retry in range(self.max_retries):
            try:
                response = self.client.messages.create(  # type: ignore
                    max_tokens=max_tokens,
                    messages=anthropic_messages,
                    model=self.model_name,
                    temperature=temperature,
                    system=system_prompt or Anthropic_NOT_GIVEN,
                    tool_choice=tool_choice_param,  # type: ignore
                    tools=tool_params,
                    extra_headers=extra_headers,
                    extra_body=extra_body,
                )
                break
            except (
                AnthropicAPIConnectionError,
                AnthropicInternalServerError,
                AnthropicRateLimitError,
                AnthropicOverloadedError,
            ) as e:
                if retry == self.max_retries - 1:
                    print(f"Failed Anthropic request after {retry + 1} retries")
                    raise e
                else:
                    print(f"Retrying LLM request: {retry + 1}/{self.max_retries}")
                    # Sleep 4-6 seconds with jitter to avoid thundering herd.
                    time.sleep(5 * random.uniform(0.8, 1.2))

        # Convert messages back to Augment format
        augment_messages = []
        assert response is not None
        for message in response.content:
            if "</invoke>" in str(message):
                warning_msg = "\n".join(
                    ["!" * 80, "WARNING: Unexpected 'invoke' in message", "!" * 80]
                )
                print(warning_msg)

            if str(type(message)) == str(AnthropicTextBlock):
                message = cast(AnthropicTextBlock, message)
                augment_messages.append(TextResult(text=message.text))
            elif str(type(message)) == str(AnthropicRedactedThinkingBlock):
                augment_messages.append(message)
            elif str(type(message)) == str(AnthropicThinkingBlock):
                message = cast(AnthropicThinkingBlock, message)
                augment_messages.append(message)
            elif str(type(message)) == str(AnthropicToolUseBlock):
                message = cast(AnthropicToolUseBlock, message)
                augment_messages.append(
                    ToolCall(
                        tool_call_id=message.id,
                        tool_name=message.name,
                        tool_input=recursively_remove_invoke_tag(message.input),
                    )
                )
            else:
                raise ValueError(f"Unknown message type: {type(message)}")

        message_metadata = {
            "raw_response": response,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                response.usage, "cache_creation_input_tokens", -1
            ),
            "cache_read_input_tokens": getattr(
                response.usage, "cache_read_input_tokens", -1
            ),
        }

        return augment_messages, message_metadata


class OpenAIDirectClient(LLMClient):
    """Use OpenAI models via first party API."""

    def __init__(self, model_name: str, max_retries=2, cot_model: bool = True):
        """Initialize the OpenAI first party client."""
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(
            api_key=api_key,
            max_retries=1,
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.cot_model = cot_model

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.

        Args:
            messages: A list of messages.
            system_prompt: A system prompt.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.

        Returns:
            A generated response.
        """
        assert thinking_tokens is None, "Not implemented for OpenAI"

        # Turn GeneralContentBlock into OpenAI message format
        openai_messages = []
        if system_prompt is not None:
            if self.cot_model:
                raise NotImplementedError("System prompt not supported for cot model")
            system_message = {"role": "system", "content": system_prompt}
            openai_messages.append(system_message)
        for idx, message_list in enumerate(messages):
            if len(message_list) > 1:
                raise ValueError("Only one entry per message supported for openai")
            augment_message = message_list[0]
            if str(type(augment_message)) == str(TextPrompt):
                augment_message = cast(TextPrompt, augment_message)
                message_content = {"type": "text", "text": augment_message.text}
                openai_message = {"role": "user", "content": [message_content]}
            elif str(type(augment_message)) == str(TextResult):
                augment_message = cast(TextResult, augment_message)
                message_content = {"type": "text", "text": augment_message.text}
                openai_message = {"role": "assistant", "content": [message_content]}
            elif str(type(augment_message)) == str(ToolCall):
                augment_message = cast(ToolCall, augment_message)
                tool_call = {
                    "type": "function",
                    "id": augment_message.tool_call_id,
                    "function": {
                        "name": augment_message.tool_name,
                        "arguments": augment_message.tool_input,
                    },
                }
                openai_message = {
                    "role": "assistant",
                    "tool_calls": [tool_call],
                }
            elif str(type(augment_message)) == str(ToolFormattedResult):
                augment_message = cast(ToolFormattedResult, augment_message)
                openai_message = {
                    "role": "tool",
                    "tool_call_id": augment_message.tool_call_id,
                    "content": augment_message.tool_output,
                }
            else:
                print(
                    f"Unknown message type: {type(augment_message)}, expected one of {str(TextPrompt)}, {str(TextResult)}, {str(ToolCall)}, {str(ToolFormattedResult)}"
                )
                raise ValueError(f"Unknown message type: {type(augment_message)}")
            openai_messages.append(openai_message)

        # Turn tool_choice into OpenAI tool_choice format
        if tool_choice is None:
            tool_choice_param = OpenAI_NOT_GIVEN
        elif tool_choice["type"] == "any":
            tool_choice_param = "required"
        elif tool_choice["type"] == "auto":
            tool_choice_param = "auto"
        elif tool_choice["type"] == "tool":
            tool_choice_param = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        else:
            raise ValueError(f"Unknown tool_choice type: {tool_choice['type']}")

        # Turn tools into OpenAI tool format
        openai_tools = []
        for tool in tools:
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            tool_def["parameters"]["strict"] = True
            openai_tool_object = {
                "type": "function",
                "function": tool_def,
            }
            openai_tools.append(openai_tool_object)

        response = None
        for retry in range(self.max_retries):
            try:
                extra_body = {}
                openai_max_tokens = max_tokens
                openai_temperature = temperature
                if self.cot_model:
                    extra_body["max_completion_tokens"] = max_tokens
                    openai_max_tokens = OpenAI_NOT_GIVEN
                    openai_temperature = OpenAI_NOT_GIVEN

                response = self.client.chat.completions.create(  # type: ignore
                    model=self.model_name,
                    messages=openai_messages,
                    temperature=openai_temperature,
                    tools=openai_tools if len(openai_tools) > 0 else OpenAI_NOT_GIVEN,
                    tool_choice=tool_choice_param,  # type: ignore
                    max_tokens=openai_max_tokens,
                    extra_body=extra_body,
                )
                break
            except (
                OpenAI_APIConnectionError,
                OpenAI_InternalServerError,
                OpenAI_RateLimitError,
            ) as e:
                if retry == self.max_retries - 1:
                    print(f"Failed OpenAI request after {retry + 1} retries")
                    raise e
                else:
                    print(f"Retrying OpenAI request: {retry + 1}/{self.max_retries}")
                    # Sleep 8-12 seconds with jitter to avoid thundering herd.
                    time.sleep(10 * random.uniform(0.8, 1.2))

        # Convert messages back to Augment format
        augment_messages = []
        assert response is not None
        openai_response_messages = response.choices
        if len(openai_response_messages) > 1:
            raise ValueError("Only one message supported for OpenAI")
        openai_response_message = openai_response_messages[0].message
        tool_calls = openai_response_message.tool_calls
        content = openai_response_message.content

        # Exactly one of tool_calls or content should be present
        if tool_calls and content:
            raise ValueError("Only one of tool_calls or content should be present")
        elif not tool_calls and not content:
            raise ValueError("Either tool_calls or content should be present")

        if tool_calls:
            if len(tool_calls) > 1:
                raise ValueError("Only one tool call supported for OpenAI")
            tool_call = tool_calls[0]
            try:
                # Parse the JSON string into a dictionary
                tool_input = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"Failed to parse tool arguments: {tool_call.function.arguments}")
                print(f"JSON parse error: {str(e)}")
                raise ValueError(f"Invalid JSON in tool arguments: {str(e)}") from e

            augment_messages.append(
                ToolCall(
                    tool_name=tool_call.function.name,
                    tool_input=tool_input,
                    tool_call_id=tool_call.id,
                )
            )
        elif content:
            augment_messages.append(TextResult(text=content))
        else:
            raise ValueError(f"Unknown message type: {openai_response_message}")

        assert response.usage is not None
        message_metadata = {
            "raw_response": response,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        return augment_messages, message_metadata


class LiteLLMClient(LLMClient):
    """使用LiteLLM访问多种模型提供商的客户端"""
    
    def __init__(
        self,
        model_name,
        max_retries=2,
        use_caching=True,#DeepSeek doesn't require to specify.
        api_base=None,
        thinking_tokens=0,
    ):
        """初始化LiteLLM客户端
        
        Args:
            model_name: 模型名称，遵循LiteLLM命名格式(如'deepseek/deepseek-v3')
            max_retries: 最大重试次数
            use_caching: 是否使用缓存
            api_key: 可选的API密钥(会覆盖环境变量)
            api_base: 可选的API基础URL
            thinking_tokens: 思考令牌数量
        """
            
        self.model_name = model_name
        self.max_retries = max_retries
        self.use_caching = use_caching
        self.api_base = api_base
        self.thinking_tokens = thinking_tokens
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        # # 设置API密钥(如果提供)
        # if api_key:
        #     provider = self._get_provider_from_model_name(model_name)
        #     if provider == "deepseek":
        #         os.environ["DEEPSEEK_API_KEY"] = api_key
        #     elif provider == "anthropic":
        #         os.environ["ANTHROPIC_API_KEY"] = api_key 
        #     elif provider == "openai":
        #         os.environ["OPENAI_API_KEY"] = api_key
    
    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """使用LiteLLM生成响应"""
        
        formatted_messages = self._format_messages(messages, system_prompt)# Its format is like SWE-Agent,a list of json.
        
        # 转换工具格式为OpenAI兼容格式
        formatted_tools = [self._convert_tool_param(tool) for tool in tools] if tools else None
        
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    kwargs = {
                        "model": self.model_name,
                        "messages": formatted_messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if formatted_tools:
                        kwargs["tools"] = formatted_tools             
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice
                    kwargs["timeout"] = 60
                    
                    # 增加兼容性处理，某些参数可能导致错误
                    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    
                    try:
                        print(f"[LiteLLM] Calling LiteLLM with model: {self.model_name}")
                        # print(f"[LiteLLM] clean_kwargs: {clean_kwargs}")
                        response = completion(**clean_kwargs)#Need to check.
                        
                        # # 输出详细的响应结构信息，帮助调试
                        # print(f"[LiteLLM] Response received, type: {type(response)}")
                        # print(f"[LiteLLM] Response dir: {dir(response)}")
                        
                        # if hasattr(response, "choices") and response.choices:
                        #     print(f"[LiteLLM] First choice type: {type(response.choices[0])}")
                        #     print(f"[LiteLLM] First choice dir: {dir(response.choices[0])}")
                            
                        #     if hasattr(response.choices[0], "message"):
                        #         message = response.choices[0].message
                        #         print(f"[LiteLLM] Message type: {type(message)}")
                        #         print(f"[LiteLLM] Message dir: {dir(message)}")
                                
                        #         # 检查是否有工具调用
                        #         if hasattr(message, "tool_calls") and message.tool_calls:
                        #             print(f"[LiteLLM] Found tool_calls, count: {len(message.tool_calls)}")
                        #             for i, tc in enumerate(message.tool_calls):
                        #                 print(f"[LiteLLM] Tool call {i} type: {type(tc)}")
                        #                 print(f"[LiteLLM] Tool call {i} dir: {dir(tc)}")
                        #         elif isinstance(message, dict) and "tool_calls" in message:
                        #             print(f"[LiteLLM] Found tool_calls in dict, count: {len(message['tool_calls'])}")
                        
                        result = self._convert_response(response)#Need to check.
                        
                        message_metadata = {
                            "raw_response": response,
                        }
                        
                        # add token count info, consistent with other clients.
                        if hasattr(response, "usage") and response.usage:
                            if hasattr(response.usage, "prompt_tokens"):
                                message_metadata["input_tokens"] = response.usage.prompt_tokens
                            elif hasattr(response.usage, "input_tokens"):
                                message_metadata["input_tokens"] = response.usage.input_tokens
                            else:
                                message_metadata["input_tokens"] = 0
                                
                            if hasattr(response.usage, "completion_tokens"):
                                message_metadata["output_tokens"] = response.usage.completion_tokens
                            elif hasattr(response.usage, "output_tokens"):
                                message_metadata["output_tokens"] = response.usage.output_tokens
                            else:
                                message_metadata["output_tokens"] = 0
                        
                        return result, message_metadata
                    except Exception as e:
                        print(f"[LiteLLM] API error: {str(e)}")
                        import traceback
                        print("[LiteLLM] Detailed error:")
                        traceback.print_exc()
                        raise
                        
                except Exception as e:
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    raise
        except Exception as e:
            print(f"[LiteLLM] Failed after {self.max_retries} retries: {str(e)}")
            error_message = f"LiteLLM API call failed: {str(e)}"
            return [TextResult(text=error_message)], {"error": str(e), "model": self.model_name}
    
    def _format_messages(self, messages: LLMMessages, system_prompt: str | None = None) -> list:
        """将内部消息格式转换为LiteLLM兼容格式"""
        formatted_messages = []
        
        # 添加系统提示
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        # 打印输入消息以便调试
        print(f"[LiteLLM] Formatting messages: {len(messages)} message groups")
        
        # 将消息组装到正确的格式
        for i, message_group in enumerate(messages):
            # 根据索引确定角色(偶数为用户，奇数为助手)
            role = "user" if i % 2 == 0 else "assistant"
            print(f"[LiteLLM] Processing message group {i}, role: {role}, blocks: {len(message_group)}")
            
            # 初始化当前角色消息的内容
            content = []
            tool_calls = []
            tool_results = []
            
            # 处理每个消息块
            for block in message_group:
                print(f"[LiteLLM] Processing block: {type(block)}")
                
                if isinstance(block, TextPrompt):
                    # 用户文本消息
                    if role == "user":
                        content.append(block.text)
                        print(f"[LiteLLM] Added user text: {block.text[:50]}...")
                        
                elif isinstance(block, TextResult):
                    # 助手文本响应
                    if role == "assistant":
                        content.append(block.text)
                        print(f"[LiteLLM] Added assistant text: {block.text[:50]}...")
                        
                elif isinstance(block, ToolCall):
                    # 助手工具调用
                    if role == "assistant":
                        try:
                            # 确保工具输入为JSON字符串
                            tool_input_str = block.tool_input
                            if not isinstance(tool_input_str, str):
                                tool_input_str = json.dumps(tool_input_str)
                                
                            tool_call = {
                                "id": block.tool_call_id or f"tool_{uuid.uuid4()}",
                                "type": "function",
                                "function": {
                                    "name": block.tool_name,
                                    "arguments": tool_input_str
                                }
                            }
                            tool_calls.append(tool_call)
                            print(f"[LiteLLM] Added tool call: {block.tool_name}")
                        except Exception as e:
                            print(f"[LiteLLM] Error formatting tool call: {str(e)}")
                            
                elif isinstance(block, ToolFormattedResult):
                    # 工具调用结果
                    tool_result = {
                        "role": "tool",
                        "content": block.tool_output,
                        "tool_call_id": block.tool_call_id
                    }
                    tool_results.append(tool_result)
                    print(f"[LiteLLM] Added tool result: {block.tool_output[:50]}...")
                    
                else:
                    print(f"[LiteLLM] Unknown block type: {type(block)}")
            
            # 根据角色和内容添加消息
            if role == "user" and content:
                # 用户消息
                user_content = content[0] if len(content) == 1 else "\n".join(content)
                formatted_messages.append({
                    "role": "user", 
                    "content": user_content
                })
                print(f"[LiteLLM] Added user message: {user_content[:50]}...")
                
            elif role == "assistant" and (content or tool_calls):
                # 助手消息
                assistant_message = {"role": "assistant"}
                
                if content:
                    assistant_content = content[0] if len(content) == 1 else "\n".join(content)
                    assistant_message["content"] = assistant_content
                    print(f"[LiteLLM] Added assistant content: {assistant_content[:50]}...")
                    
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                    print(f"[LiteLLM] Added {len(tool_calls)} tool calls to assistant message")
                    
                formatted_messages.append(assistant_message)
            
            # 添加工具结果（这些应该直接添加到消息列表中，而不是作为角色消息的一部分）
            for tool_result in tool_results:
                formatted_messages.append(tool_result)
        
        print(f"[LiteLLM] Formatted {len(formatted_messages)} messages")
        return formatted_messages
    
    def _convert_tool_param(self, tool: ToolParam) -> dict:
        """将内部工具参数转换为LiteLLM兼容格式"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
        }
    
    def _convert_response(self, response) -> list[AssistantContentBlock]:
        """将LiteLLM响应转换为内部格式"""
        result = []
        
        try:
            choice = response.choices[0]
            message = choice.message
            
            # 打印详细响应信息以便调试
            print(f"[LiteLLM] Processing response: {message}")
            if hasattr(message, "tool_calls"):
                print(f"[LiteLLM] Tool calls found (attribute): {message.tool_calls}")
            elif isinstance(message, dict) and "tool_calls" in message:
                print(f"[LiteLLM] Tool calls found (dict): {message['tool_calls']}")
            
            # 处理文本响应 - 首先处理，避免后续工具调用失败时没有任何输出
            content = None
            if hasattr(message, "content") and message.content is not None:
                content = message.content
            elif isinstance(message, dict) and "content" in message:
                content = message["content"]
                
            if content is not None:
                result.append(TextResult(text=content))
            
            # 处理工具调用 - 兼容不同的响应格式
            tool_calls = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = message.tool_calls
            elif isinstance(message, dict) and "tool_calls" in message:
                tool_calls = message["tool_calls"]
                
            if tool_calls:
                for tool_call in tool_calls:
                    try:
                        # 打印工具调用详情以便调试
                        print(f"[LiteLLM] Processing tool call: {tool_call}")
                        print(f"[LiteLLM] Tool call type: {type(tool_call)}")
                        
                        # 获取工具类型，兼容不同格式
                        tool_type = None
                        if isinstance(tool_call, dict) and "type" in tool_call:
                            tool_type = tool_call["type"]
                        elif hasattr(tool_call, "type"):
                            tool_type = tool_call.type
                        # 对于没有明确类型的，假设为function
                        else:
                            tool_type = "function"
                            
                        print(f"[LiteLLM] Tool type identified: {tool_type}")
                        
                        # 处理函数类型工具调用
                        if tool_type == "function" or tool_call is not None:
                            # 获取函数信息，兼容不同格式
                            function = None
                            if isinstance(tool_call, dict) and "function" in tool_call:
                                function = tool_call["function"]
                            elif hasattr(tool_call, "function"):
                                function = tool_call.function
                            # 如果没有function字段但有function属性
                            elif hasattr(tool_call, "name") and hasattr(tool_call, "arguments"):
                                function = tool_call
                            
                            # 获取函数名称
                            function_name = None
                            if isinstance(function, dict) and "name" in function:
                                function_name = function["name"]
                            elif hasattr(function, "name"):
                                function_name = function.name
                            elif hasattr(tool_call, "name"):  # 某些模型可能直接在工具调用中包含名称
                                function_name = tool_call.name
                                
                            # 获取参数
                            arguments = None
                            if isinstance(function, dict) and "arguments" in function:
                                arguments = function["arguments"]
                            elif hasattr(function, "arguments"):
                                arguments = function.arguments
                            elif hasattr(tool_call, "arguments"):  # 某些模型可能直接在工具调用中包含参数
                                arguments = tool_call.arguments
                            
                            # 获取工具ID
                            tool_id = None
                            if isinstance(tool_call, dict) and "id" in tool_call:
                                tool_id = tool_call["id"]
                            elif hasattr(tool_call, "id"):
                                tool_id = tool_call.id
                            elif hasattr(tool_call, "tool_call_id"):  # 某些模型可能使用不同的字段名
                                tool_id = tool_call.tool_call_id
                            # 如果没有ID，生成一个唯一ID
                            else:
                                tool_id = f"tool_{uuid.uuid4()}"
                            
                            print(f"[LiteLLM] Function details: name={function_name}, id={tool_id}")
                            print(f"[LiteLLM] Arguments: {arguments}")
                            
                            # 解析参数
                            tool_input = None
                            try:
                                if isinstance(arguments, str):
                                    # 尝试解析JSON
                                    try:
                                        tool_input = json.loads(arguments) 
                                    except json.JSONDecodeError:
                                        # 如果不是有效的JSON，保留原始字符串
                                        tool_input = arguments
                                else:
                                    tool_input = arguments
                            except Exception as e:
                                print(f"[LiteLLM] Error parsing arguments: {e}")
                                tool_input = arguments
                            
                            # 添加工具调用
                            if function_name:  # 只需要函数名，其他可以生成或默认
                                print(f"[LiteLLM] Adding tool call: {function_name}")
                                result.append(
                                    ToolCall(
                                        tool_call_id=tool_id,
                                        tool_name=function_name,
                                        tool_input=tool_input
                                    )
                                )
                    except Exception as e:
                        print(f"[LiteLLM] Error processing tool call: {e}")
                        import traceback
                        traceback.print_exc()
                        # 继续处理下一个工具调用
            
            # 如果没有任何内容，至少返回一个空文本结果
            if not result:
                print("[LiteLLM] No content or tool calls found, returning empty text result")
                result.append(TextResult(text=""))
                
        except Exception as e:
            print(f"[LiteLLM] Error in _convert_response: {str(e)}")
            import traceback
            traceback.print_exc()
            result.append(TextResult(text=f"Error processing response: {str(e)}"))
        
        return result


def get_client(client_name: str, **kwargs) -> LLMClient:
    """Get a client for a given client name."""
    if client_name == "anthropic-direct":
        return AnthropicDirectClient(**kwargs)
    elif client_name == "openai-direct":
        return OpenAIDirectClient(**kwargs)
    elif client_name == "litellm":
        model_name = kwargs.get("model_name")
        if model_name is None:
            raise ValueError("Must specify model_name parameter for LiteLLM client")

        if model_name == "deepseekv3" or model_name == "deepseek-v3":
            kwargs["model_name"] = "deepseek/deepseek-chat" 
        elif "deepseek" in model_name.lower() and not model_name.startswith("deepseek/"):
            kwargs["model_name"] = f"deepseek/{model_name}"
        elif "deepseek" not in model_name.lower():
            raise ValueError(f"Unknown model name: {model_name}, only supports deepseek models")
        # Todo Support other models......
        return LiteLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown client name: {client_name}")
