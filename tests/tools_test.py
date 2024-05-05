import pytest
import json

from unittest.mock import Mock
from llm_easy_tools import ToolBox, ToolResult, llm_function
from pydantic import BaseModel, Field, ValidationError
from typing import Any
from openai.types.chat.chat_completion import ChatCompletionMessage, ChatCompletion

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

def test_toolbox_init():
    toolbox = ToolBox()
    assert toolbox._tool_registry == {}
    assert toolbox.tool_schemas() == []
    assert toolbox.case_insensitive == False
    assert toolbox.fix_json_args == True

def test_register_toolset():
    tool_manager = ToolBox()

    # Test the normal case
    tool_manager.register_toolset(tool)

    toolset = tool_manager.get_toolset('TestTool')
    assert toolset == tool

    assert 'tool_method' in tool_manager._tool_registry
    assert 'additional_tool_method' in tool_manager._tool_registry
    assert 'User' in tool_manager._tool_registry
    assert 'SomeClass' not in tool_manager._tool_registry
    assert 'short_address' in tool_manager._tool_registry
    assert '_private_tool_method' not in tool_manager._tool_registry

    # Test for Exception when a Toolset with same key is being registered
    with pytest.raises(Exception) as exception_info:
        tool_manager.register_toolset(tool)

    assert str(exception_info.value) == 'A toolset with key TestTool already exists.'


class FunctionCallMock:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments



class UserDetail(BaseModel):
    name: str
    age: int

def mk_chat_with_tool_call(name, args):
    message = ChatCompletionMessage(
        role="assistant",
        tool_calls=[
            {
                "id": 'A',
                "type": 'function',
                "function": {
                    "arguments": json.dumps(args),
                    "name": name
                }
            }
        ]
    )
    chat_completion = ChatCompletion(
        id='A',
        created=0,
        model='A',
        choices=[{'finish_reason': 'stop', 'index': 0, 'message': message}],
        object='chat.completion'
    )
    return chat_completion


def test_process_response():
    # too much mocking in this test
    toolbox = ToolBox()
    toolbox.register_function(UserDetail)
    original_user = UserDetail(name="John", age=21)
    response = mk_chat_with_tool_call("UserDetail", original_user.model_dump())
    results = toolbox.process_response(response)
    assert len(results) == 1
    assert results[0].output == original_user
    message = results[0].to_message()
    assert message['role'] == 'tool'
    assert message['tool_call_id'] == 'A'
    assert message['name'] == 'UserDetail'
    assert message['content'] == 'UserDetail created'


# Define the test cases
def test_register_tool():

    def example_tool(name: str):
        print(f'Running test tool with name param: "{name}"')


    toolbox = ToolBox()

    # Test with correct parameters
    toolbox.register_function(example_tool)
    assert 'example_tool' in toolbox._tool_registry
    function_info = toolbox._tool_registry['example_tool']
    assert function_info["function"] == example_tool
    assert len(toolbox.tool_schemas()) == 1

    # Test with function with no parameters
    def no_parameters(): pass
    toolbox.register_function(no_parameters)
    assert 'no_parameters' in toolbox._tool_registry

    @llm_function("good_name")
    def bad_name_tool(name: str):
        print('Running bad_name_tool')

    toolbox.register_function(bad_name_tool)
    assert 'good_name' in toolbox._tool_registry
    function_info = toolbox._tool_registry['good_name']
    assert function_info["function"] == bad_name_tool

def test_register_model():

    class Tool(BaseModel):
        name: str

    class WikiSearch(BaseModel):
        query: str

    toolbox = ToolBox()
    toolbox.register_function(Tool)
    x = toolbox._tool_registry['Tool']['function'](name="Some name")
    assert x.name == 'Some name'

    assert len(toolbox.tool_schemas()) == 1
    assert toolbox.tool_schemas()[0]['function']['name'] == 'Tool'

    toolbox.register_function(WikiSearch)
    assert len(toolbox.tool_schemas()) == 2

    assert toolbox.get_tool_schema('WikiSearch')['function']['name'] == 'WikiSearch'

def test_prefixing():

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")

    def example_tool(name: str):
        return 'test tool result'

    toolbox = ToolBox()
    toolbox.register_function(example_tool)
    tool_schemas = toolbox.tool_schemas(prefix_class=Reflection)
    assert len(tool_schemas) == 1
    function_schema = tool_schemas[0]['function']
    first_param_name = list(function_schema['parameters']['properties'].keys())[0]
    assert first_param_name == 'relevancy'
    assert function_schema['name'] == 'Reflection_and_example_tool'


def test_process_response_with_prefixing():

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")

    toolbox = ToolBox()
    toolbox.register_toolset(tool)
    prefixed_name = 'Reflection_and_tool_method'
    response = mk_chat_with_tool_call(prefixed_name, {'arg': 2})
    results = toolbox.process_response(response, prefix_class=Reflection)
    assert results[0].output == 'executed tool_method with param: 2'
    assert results[0].prefix is None
    assert len(results[0].soft_errors) == 1
    args = {'arg': 2}
    args['relevancy'] = 'very good'
    response = mk_chat_with_tool_call(prefixed_name, args)
    results = toolbox.process_response(response, prefix_class=Reflection)
    assert results[0].output == 'executed tool_method with param: 2'
    assert isinstance(results[0].prefix, Reflection)

pytest.main()
