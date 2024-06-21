import pytest
import json
from time import sleep, time

from unittest.mock import Mock
from pydantic import BaseModel, Field, ValidationError
from typing import Any
from openai.types.chat.chat_completion import ChatCompletionMessage, ChatCompletion, Choice
from openai.types.chat.chat_completion_message_tool_call   import ChatCompletionMessageToolCall, Function

from llm_easy_tools.processor import process_response, process_tool_call, ToolResult, _extract_prefix_unpacked
from llm_easy_tools import llm_function
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def mk_tool_call(name, args):
    arguments = json.dumps(args)
    return ChatCompletionMessageToolCall(id='A', function=Function(name=name, arguments=arguments), type='function')

def mk_tool_call_jason(name, args):
    return ChatCompletionMessageToolCall(id='A', function=Function(name=name, arguments=args), type='function')

def mk_chat_completion(tool_calls):
    return ChatCompletion(
        id='A',
        created=0,
        model='gpt-3.5-turbo',
        object='chat.completion',
        choices=[
            Choice(
                finish_reason='stop',
                index=0,
                message=ChatCompletionMessage(role='assistant', tool_calls=tool_calls))
        ]
    )


def test_process_methods():
    class TestTool:

        def tool_method(self, arg: int) -> str:
            return f'executed tool_method with param: {arg}'

        def no_output(self, arg: int):
            pass

        def failing_method(self, arg: int) -> str:
            raise Exception('Some exception')


    tool = TestTool()

    tool_call = mk_tool_call("tool_method", {"arg": 2})
    result = process_tool_call(tool_call, [tool.tool_method])
    assert isinstance(result, ToolResult)
    assert result.output == 'executed tool_method with param: 2'

    tool_call = mk_tool_call("failing_method", {"arg": 2})
    result = process_tool_call(tool_call, [tool.failing_method])
    assert isinstance(result, ToolResult)
    assert "Some exception" in str(result.error)
    message = result.to_message()
    assert "Some exception" in message['content']

    tool_call = mk_tool_call("no_output", {"arg": 2})
    result = process_tool_call(tool_call, [tool.no_output])
    assert isinstance(result, ToolResult)
    message = result.to_message()
    assert message['content'] == ''

def test_process_complex():

    class Address(BaseModel):
        street: str
        city: str

    class Company(BaseModel):
        name: str
        speciality: str
        address: Address


    def print_companies(companies: list[Company]):
        return companies

    company_list = [{
        'address': {'city': 'Metropolis', 'street': '150 Futura Plaza'},
        'name': 'Aether Innovations',
        'speciality': 'sustainable energy solutions'
    }]

    tool_call = mk_tool_call("print_companies", {"companies": company_list})
    result = process_tool_call(tool_call, [print_companies])
    assert isinstance(result, ToolResult)
    assert isinstance(result.output, list)
    assert isinstance(result.output[0], Company)

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
    tool_call = mk_tool_call_jason("UserDetail", json_data)
    result = process_tool_call(tool_call, [UserDetail])
    assert result.output == original_user
    assert len(result.soft_errors) > 0

    result = process_tool_call(tool_call, [UserDetail], fix_json_args=False)
    assert isinstance(result.error, json.decoder.JSONDecodeError)

    response = mk_chat_completion([tool_call])
    results = process_response(response, [UserDetail])
    assert results[0].output == original_user
    assert len(results[0].soft_errors) > 0

    results = process_response(response, [UserDetail], fix_json_args=False)
    assert isinstance(results[0].error, json.decoder.JSONDecodeError)

def test_list_in_string_fix():
    class User(BaseModel):
        names: list[str]

    tool_call = mk_tool_call("User", {"names": "John, Doe"})
    result = process_tool_call(tool_call, [User])
    assert result.output.names == ["John", "Doe"]
    assert len(result.soft_errors) > 0

    result = process_tool_call(tool_call, [User], fix_json_args=False)
    assert isinstance(result.error, ValidationError)

def test_case_insensitivity():
    class User(BaseModel):
        name: str
        city: str

    response = mk_chat_completion([mk_tool_call("user", {"name": "John", "city": "Metropolis"})])
    results = process_response(response, [User], case_insensitive=True)
    assert results[0].output == User(name="John", city="Metropolis")

def test_parallel_tools():

    class CounterClass:
        def __init__(self):
            self.counter = 0

        def increment_counter(self):
            self.counter += 1
            sleep(1)  # Increased sleep time to 1 second

    counter = CounterClass()
    tool_call = mk_tool_call("increment_counter", {})
    response = mk_chat_completion([tool_call] * 10)

    executor = ThreadPoolExecutor()
    start_time = time()
    results = process_response(response, [counter.increment_counter], executor=executor)
    end_time = time()

    assert results[9].error is None

    time_taken = end_time - start_time
    assert counter.counter == 10
    assert time_taken <= 2, f"Expected processing time to be less than or equal to 2 seconds, but was {time_taken}"
