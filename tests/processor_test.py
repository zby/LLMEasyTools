import pytest
import json

from unittest.mock import Mock
from llm_easy_tools import llm_function
from pydantic import BaseModel, Field, ValidationError
from typing import Any
from openai.types.chat.chat_completion import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion_message_tool_call   import ChatCompletionMessageToolCall, Function

from llm_easy_tools.processor import process_response, process_tool_call, ToolResult, _extract_prefix_unpacked

class TestTool:

    class SomeClass(BaseModel):
        value: int

    @llm_function()
    def tool_method(self, arg: int) -> str:
        return f'executed tool_method with param: {arg}'

    @llm_function()
    def additional_tool_method(self, arg: int) -> str:
        return f'executed additional_tool_method with param: {arg}'

    def _private_tool_method(self, arg: int) -> str:
        return str(arg.value * 4)

    @llm_function()
    class User(BaseModel):
        name: str
        age: int

    @llm_function('short_address')
    class Address(BaseModel):
        city: str
        street: str

    @llm_function()
    def no_output(self, arg: int):
        pass


    @llm_function()
    def failing_method(self, arg: int) -> str:
        raise Exception('Some exception')


tool = TestTool()


def mk_tool_call(name, arguments):
    return ChatCompletionMessageToolCall(id='A', function=Function(name=name, arguments=arguments), type='function')

def test_process():
    tool_call = mk_tool_call("tool_method", json.dumps({"arg": 2}))
    result = process_tool_call(tool_call, [tool.tool_method])
    assert isinstance(result, ToolResult)
    assert result.output == 'executed tool_method with param: 2'

    tool_call = mk_tool_call("failing_method", json.dumps({"arg": 2}))
    result = process_tool_call(tool_call, [tool.failing_method])
    assert isinstance(result, ToolResult)
    assert "Some exception" in result.error
    message = result.to_message()
    assert "Some exception" in message['content']

    tool_call = mk_tool_call("no_output", json.dumps({"arg": 2}))
    result = process_tool_call(tool_call, [tool.no_output])
    assert isinstance(result, ToolResult)
    message = result.to_message()
    assert message['content'] == ''


def test_prefixing():

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")

    args = { 'relevancy': 'good', 'name': 'hammer'}
    prefix = _extract_prefix_unpacked(args, Reflection)
    assert isinstance(prefix, Reflection)
    assert 'reflection' not in args # prefix params extracted


def test_json_fix():

    class UserDetail(BaseModel):
        name: str
        age: int

    original_user = UserDetail(name="John", age=21)
    json_data = json.dumps(original_user.model_dump())
    json_data = json_data[:-1]
    json_data = json_data + ',}'
    tool_call = mk_tool_call("UserDetail", json_data)
    result = process_tool_call(tool_call, [UserDetail])
    assert result.output == original_user
    assert len(result.soft_errors) > 0

    result = process_tool_call(tool_call, [UserDetail], fix_json_args=False)
    assert 'json.decoder.JSONDecodeError' in result.error

