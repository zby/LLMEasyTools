import pytest
import json

from unittest.mock import Mock
from llm_easy_tools import ToolBox, SchemaGenerator, external_function, extraction_model
from pydantic import BaseModel, Field, ValidationError
from typing import Any


class ToolParam(BaseModel):
    value: int


class AdditionalToolParam(BaseModel):
    value: int

class TestTool:

    class SomeClass(BaseModel):
        value: int

    @external_function()
    def tool_method(self, arg: ToolParam) -> str:
        return f'executed tool_method with param: {arg}'

    @external_function()
    def additional_tool_method(self, arg: AdditionalToolParam) -> str:
        return f'executed additional_tool_method with param: {arg}'

    def _private_tool_method(self, arg: AdditionalToolParam) -> str:
        return str(arg.value * 4)

    @extraction_model()
    class User(BaseModel):
        name: str
        age: int

    @extraction_model('address')
    class Address(BaseModel):
        city: str
        street: str


tool = TestTool()

def test_toolbox_init():
    toolbox = ToolBox()
    assert toolbox.strict == True
    assert toolbox.tool_registry == {}
    assert toolbox.name_mappings == []
    assert toolbox.tool_schemas() == []
    assert isinstance(toolbox.generator, SchemaGenerator)

def test_register_toolset():
    tool_manager = ToolBox()

    # Test the normal case
    tool_manager.register_toolset(tool)

    assert 'TestTool' in tool_manager.tool_sets
    assert 'tool_method' in tool_manager.tool_registry
    assert 'additional_tool_method' in tool_manager.tool_registry
    assert 'User' in tool_manager.tool_registry
    assert 'SomeClass' not in tool_manager.tool_registry
    assert 'Address' in tool_manager.tool_registry
    assert tool_manager.schema_name_to_func('address') == 'Address'
    assert '_private_tool_method' not in tool_manager.tool_registry

    # Test for Exception when a Toolset with same key is being registered
    with pytest.raises(Exception) as exception_info:
        tool_manager.register_toolset(tool)

    assert str(exception_info.value) == 'A toolset with key TestTool already exists.'

def test_toolbox_from_object():
    toolbox = ToolBox.toolbox_from_object(tool)
    assert "tool_method" in toolbox.tool_registry
    assert len(toolbox.tool_registry) == 4
    assert len(toolbox.tool_schemas()) == 4

def test_schema_name_to_func():
    toolbox = ToolBox(name_mappings=[("tool_method", "TestTool")])
    assert toolbox.schema_name_to_func("TestTool") == "tool_method"
    assert toolbox.schema_name_to_func("NotFound") == "NotFound"


class FunctionCallMock:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

def test_process():
    toolbox = ToolBox.toolbox_from_object(tool)
    function_call = FunctionCallMock(name="tool_method", arguments=json.dumps(ToolParam(value=2).model_dump()))
    result = toolbox.process_function(function_call)
    assert result == 'executed tool_method with param: value=2'

    toolbox = ToolBox.toolbox_from_object(tool, name_mappings=[("additional_tool_method", "custom_name")])
    function_call = FunctionCallMock(name="custom_name", arguments=json.dumps(ToolParam(value=3).model_dump()))
    result = toolbox.process_function(function_call)
    assert result == "executed additional_tool_method with param: value=3"

    # Test with unknown function call name
    with pytest.raises(ValueError):
        function_call = FunctionCallMock(name="unknown_name", arguments=json.dumps(ToolParam(value=3).model_dump()))
        toolbox.process_function(function_call)


class UserDetail(BaseModel):
    name: str
    age: int

def test_process_with_identity():
    toolbox = ToolBox()
    toolbox.register_model(UserDetail)
    assert "UserDetail" in toolbox.tool_registry
    original_user = UserDetail(name="John", age=21)
    function_call = FunctionCallMock(name="UserDetail", arguments=json.dumps(original_user.model_dump()))
    result = toolbox.process_function(function_call)
    assert result == original_user

def process_response():
    toolbox = ToolBox()
    toolbox.register_function(UserDetail)
    original_user = UserDetail(name="John", age=21)
    function_call = FunctionCallMock(name="UserDetail", arguments=json.dumps(original_user.model_dump()))
    response = Mock(choices=[Mock(message=Mock(tool_calls=[Mock(function=function_call)]))])
    results = toolbox.process_response(response)
    assert len(results) == 1
    assert results[0] == original_user


# Define the test cases
def test_register_tool():
    class Tool(BaseModel):
        name: str

    def example_tool(tool: Tool):
        print('Running test tool')

    @external_function("good_name")
    def bad_name_tool(tool: Tool):
        print('Running bad_name_tool')

    toolbox = ToolBox()

    # Test with correct parameters
    toolbox.register_function(example_tool)
    assert 'example_tool' in toolbox.tool_registry
    function_info = toolbox.tool_registry['example_tool']
    assert function_info["function"] == example_tool
    assert function_info["param_class"] == Tool
    assert len(toolbox.tool_schemas()) == 1
    assert len(toolbox.function_schemas()) == 1
    assert toolbox.tool_schemas()[0]['function']['name'] == 'example_tool'
    assert toolbox.function_schemas()[0]['name'] == 'example_tool'

    # Test with function with more than one parameter
    with pytest.raises(TypeError):
        def two_parameters(a, b): pass
        toolbox.register_function(two_parameters)

    # Test with function with no parameters
    with pytest.raises(TypeError):
        def no_parameters(): pass
        toolbox.register_function(no_parameters)

    # Test with function having a parameter which isn't subclass of BaseModel
    with pytest.raises(TypeError):
        def wrong_parameter(a: Any): pass
        toolbox.register_function(wrong_parameter)

    toolbox.register_function(bad_name_tool)
    assert 'bad_name_tool' in toolbox.tool_registry
    function_info = toolbox.tool_registry['bad_name_tool']
    assert function_info["function"] == bad_name_tool
    assert function_info["param_class"] == Tool
    assert toolbox.schema_name_to_func('good_name') == 'bad_name_tool'

def test_register_model():
    class Tool(BaseModel):
        name: str

    class WikiSearch(BaseModel):
        query: str

    toolbox = ToolBox()
    toolbox.register_model(Tool)

    identity_function = toolbox.tool_registry['Tool']["function"]
    assert callable(identity_function)
    assert identity_function(Tool(name="test")) == Tool(name="test")
    assert toolbox.tool_registry['Tool']["param_class"] is Tool
    assert len(toolbox.tool_schemas()) == 1
    assert toolbox.tool_schemas()[0]['function']['name'] == 'tool' # by default case_insensitive

    toolbox.register_model(WikiSearch)
    identity_function = toolbox.tool_registry['WikiSearch']["function"]
    assert callable(identity_function)
    assert identity_function(WikiSearch(query="test")) == WikiSearch(query="test")
    assert toolbox.tool_registry['WikiSearch']["param_class"] is WikiSearch
    assert len(toolbox.tool_schemas()) == 2
    assert toolbox.tool_schemas()[1]['function']['name'] == 'wikisearch'

    assert toolbox.get_tool_schema('WikiSearch')['function']['name'] == 'wikisearch'
    assert toolbox.get_tool_schema('WikiSearch')['type'] == 'function'

def test_prefixing():
    class Tool(BaseModel):
        name: str

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")

    def example_tool(tool: Tool):
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
    prefixed_name = 'reflection_and_tool_method'
    no_reflection_function_call = FunctionCallMock(name=prefixed_name, arguments=json.dumps(ToolParam(value=2).model_dump()))
    with pytest.raises(Exception) as exception_info:
        toolbox.process_function(no_reflection_function_call, prefix_class=Reflection)
    assert isinstance(exception_info.value, ValidationError)
    args = ToolParam(value=2).model_dump()
    args['relevancy'] = 'very good'
    function_call = FunctionCallMock(name=prefixed_name, arguments=json.dumps(args))
    result = toolbox.process_function(function_call, prefix_class=Reflection)
    assert result == 'executed tool_method with param: value=2'
    assert isinstance(toolbox.prefix, Reflection)


pytest.main()
