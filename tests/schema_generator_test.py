import pytest

from typing import List, Optional
from pydantic import BaseModel, Field

from llm_easy_tools import SchemaGenerator


class Foo(BaseModel):
    count: int
    size: Optional[float] = None


class Bar(BaseModel):
    """Some Bar"""
    apple: str = Field(description="The apple")
    banana: str = Field(description="The banana")

def simple_function(param: Foo):
    """simple function does something"""
    pass


def simple_function_no_docstring(param: Bar):
    pass

def function_with_docstring(param: Bar):
    "function with docstring"
    pass


def test_initialization():
    # Test default initialization
    generator = SchemaGenerator()
    assert generator.strict
    custom_name_mappings = [("func", "custom_func")]
    generator = SchemaGenerator(strict=False, name_mappings=custom_name_mappings)
    assert not generator.strict
    assert generator.name_mappings == custom_name_mappings
    assert generator.func_name_to_schema('func') == 'custom_func'
    assert generator.schema_name_to_func('custom_func') == 'func'

def test_function_schema():
    generator = SchemaGenerator()

    function_schema = generator.function_schema(simple_function)
    assert function_schema['name'] == 'simple_function'
    assert function_schema['description'] == 'simple function does something'
    params_schema = function_schema['parameters']
    assert len(params_schema['properties']) == 2
    assert params_schema['type'] == "object"
    assert params_schema['properties']['count']['type'] == "integer"
    assert 'size' in params_schema['properties']
    assert 'title' not in params_schema
    assert 'title' not in params_schema['properties']['count']
    assert 'description' not in params_schema

def test_empty_model_schemas():
    class EmptyModel(BaseModel):
        pass

    class EmptyModelWithDoc(BaseModel):
        """
        This is an empty model with a docstring.
        """
        pass

    def function_with_doc(empty_model: EmptyModel):
        """
        This function has a docstring and takes an empty model as parameter.

        :return: This is a dummy function
        """
        pass

    def function_no_doc(empty_model: EmptyModel):
        pass

    def function_no_doc_with_doc_model(empty_model: EmptyModelWithDoc):
        pass

    generator = SchemaGenerator()

    result = generator.function_schema(function_with_doc)
    # Function with docstring and EmptyModel as parameter
    assert result['name'] == 'function_with_doc'
    assert result['description'] == "This function has a docstring and takes an empty model as parameter."
    assert 'parameters' not in result # "Omitting parameters defines a function with an empty parameter list."

    # Function without docstring and EmptyModel as parameter
    result = generator.function_schema(function_no_doc)
    assert result['name'] == 'function_no_doc'
    assert result['description'] == ''
    assert 'parameters' not in result

    # Function without docstring and EmptyModelWithDoc as parameter
    result = generator.function_schema(function_no_doc_with_doc_model)
    assert result['name'] == 'function_no_doc_with_doc_model'
    assert result['description'] == 'This is an empty model with a docstring.'
    assert 'parameters' not in result

def test_nested():
    class Spam(BaseModel):
        foo: Foo
        bars: List[Bar]
    def nested_structure_function(param: Spam):
        """spams everything"""
        pass
    generator = SchemaGenerator()

    function_schema = generator.function_schema(nested_structure_function)
    assert function_schema['name'] == 'nested_structure_function'
    assert function_schema['description'] == 'spams everything'
    assert len(function_schema['parameters']['properties']) == 2



def test_exceptions():
    generator = SchemaGenerator()

    def function_wrong_params(param: str):
        pass

    with pytest.raises(ValueError) as exc_info:
        generator.function_schema(function_with_docstring)

    # Optional: Check the message of the exception
    assert str(exc_info.value) == f"Both function '{function_with_docstring.__name__}' and the parameter class 'Bar' have descriptions"

    with pytest.raises(TypeError) as exc_info:
        generator.function_schema(function_wrong_params)


def test_methods():
    class ExampleClass:
        def simple_method(self, param: Foo):
            """simple method does something"""
            pass

    generator = SchemaGenerator()
    example_object = ExampleClass()

    function_schema = generator.function_schema(example_object.simple_method)
    assert function_schema['name'] == 'simple_method'
    assert function_schema['description'] == 'simple method does something'
    params_schema = function_schema['parameters']
    assert len(params_schema['properties']) == 2

def test_tools():
    generator = SchemaGenerator(name_mappings=[('simple_function_no_docstring', 'new_custom_name')])
    tools = generator.generate_tools(simple_function, simple_function_no_docstring)
    assert len(tools) == 2
    assert tools[0]['function']['name'] == 'simple_function'
    assert tools[1]['function']['name'] == 'new_custom_name'
