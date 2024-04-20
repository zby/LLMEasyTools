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


tool = TestTool()

def test_toolbox_init():
    toolbox = ToolBox()
    assert toolbox.tool_registry == {}
    assert toolbox.tool_schemas() == []
    assert toolbox.case_insensitive == False
    assert toolbox.fix_json_args == True

def test_register_toolset():
    tool_manager = ToolBox()

    # Test the normal case
    tool_manager.register_toolset(tool)

    assert 'TestTool' in tool_manager.tool_sets
    assert 'tool_method' in tool_manager.tool_registry
    assert 'additional_tool_method' in tool_manager.tool_registry
    assert 'User' in tool_manager.tool_registry
    assert 'SomeClass' not in tool_manager.tool_registry
    assert 'short_address' in tool_manager.tool_registry
    assert '_private_tool_method' not in tool_manager.tool_registry

    # Test for Exception when a Toolset with same key is being registered
    with pytest.raises(Exception) as exception_info:
        tool_manager.register_toolset(tool)

    assert str(exception_info.value) == 'A toolset with key TestTool already exists.'


class FunctionCallMock:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

def test_process():
    toolbox = ToolBox()
    toolbox.register_toolset(tool)
    function_call = FunctionCallMock(name="tool_method", arguments=json.dumps({"arg": 2}))
    result = toolbox.process_function(function_call, '')
    assert isinstance(result, ToolResult)
    assert result.output == 'executed tool_method with param: 2'

    # Test with unknown function call name
    with pytest.raises(KeyError):
        function_call = FunctionCallMock(name="unknown", arguments=json.dumps({'arg': 3}))
        toolbox.process_function(function_call, '')


class UserDetail(BaseModel):
    name: str
    age: int

def test_process_model():
    toolbox = ToolBox()
    toolbox.register_model(UserDetail)
    assert "UserDetail" in toolbox.tool_registry
    original_user = UserDetail(name="John", age=21)
    function_call = FunctionCallMock(name="UserDetail", arguments=json.dumps(original_user.model_dump()))
    result = toolbox.process_function(function_call, '')
    assert result.model == original_user

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
    toolbox.register_model(UserDetail)
    original_user = UserDetail(name="John", age=21)
    response = mk_chat_with_tool_call("UserDetail", original_user.model_dump())
    results = toolbox.process_response(response)
    assert len(results) == 1
    assert results[0].model == original_user


# Define the test cases
def test_register_tool():
    class Tool(BaseModel):
        name: str

    def example_tool(name: str):
        print(f'Running test tool with name param: "{name}"')


    toolbox = ToolBox()

    # Test with correct parameters
    toolbox.register_function(example_tool)
    assert 'example_tool' in toolbox.tool_registry
    function_info = toolbox.tool_registry['example_tool']
    assert function_info["function"] == example_tool
    assert len(toolbox.tool_schemas()) == 1

    # Test with function with no parameters
    def no_parameters(): pass
    toolbox.register_function(no_parameters)
    assert 'no_parameters' in toolbox.tool_registry

    @llm_function("good_name")
    def bad_name_tool(name: str):
        print('Running bad_name_tool')

    toolbox.register_function(bad_name_tool)
    assert 'good_name' in toolbox.tool_registry
    function_info = toolbox.tool_registry['good_name']
    assert function_info["function"] == bad_name_tool

def test_register_model():
    class Tool(BaseModel):
        name: str

    class WikiSearch(BaseModel):
        query: str

    toolbox = ToolBox()
    toolbox.register_model(Tool)

    assert toolbox.tool_registry['Tool']["model_class"] is Tool
    assert len(toolbox.tool_schemas()) == 1
    assert toolbox.tool_schemas()[0]['function']['name'] == 'Tool'

    toolbox.register_model(WikiSearch)
    assert toolbox.tool_registry['WikiSearch']["model_class"] is WikiSearch
    assert len(toolbox.tool_schemas()) == 2

    assert toolbox.get_tool_schema('WikiSearch')['name'] == 'WikiSearch'

def test_prefixing():
    class Tool(BaseModel):
        name: str

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

    args = { 'relevancy': 'good', 'name': 'hammer'}
    prefix = toolbox._extract_prefix_unpacked(args, Reflection)
    assert isinstance(prefix, Reflection)
    assert 'reflection' not in args # prefix params extracted


def test_process_function_with_prefixing():
    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")

    toolbox = ToolBox()
    toolbox.register_toolset(tool)
    prefixed_name = 'Reflection_and_tool_method'
    no_reflection_function_call = FunctionCallMock(name=prefixed_name, arguments=json.dumps({'arg': 2}))
    with pytest.raises(Exception) as exception_info:
        toolbox.process_function(no_reflection_function_call, '', prefix_class=Reflection)
    assert isinstance(exception_info.value, ValidationError)
    args = {'arg': 2}
    args['relevancy'] = 'very good'
    function_call = FunctionCallMock(name=prefixed_name, arguments=json.dumps(args))
    result = toolbox.process_function(function_call, '', prefix_class=Reflection)
    assert result.output == 'executed tool_method with param: 2'
    assert isinstance(toolbox.prefix, Reflection)

def test_json_fix():
    toolbox = ToolBox()
    toolbox.register_model(UserDetail)
    original_user = UserDetail(name="John", age=21)
    json_data = json.dumps(original_user.model_dump())
    json_data = json_data[:-1]
    json_data = json_data + ',}'
    function_call = FunctionCallMock(name="UserDetail", arguments=json_data)
    result = toolbox.process_function(function_call, '')
    assert result.model == original_user

    toolbox.fix_json_args = False
    result = toolbox.process_function(function_call, '')
    assert 'json.decoder.JSONDecodeError' in result.error

pytest.main()
