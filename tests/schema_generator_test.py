import pytest

from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator

from llm_easy_tools import SchemaGenerator

from pprint import pprint

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

    # Function without docstring and EmptyModelWithDoc as parameter
    result = generator.function_schema(function_no_doc_with_doc_model)
    assert result['name'] == 'function_no_doc_with_doc_model'
    assert result['description'] == 'This is an empty model with a docstring.'
    assert 'parameters' not in result

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


def test_merge_schemas():

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")
        next_actions_plan: str = Field(..., description="What you plan to do next and why")

    generator = SchemaGenerator()
    function_schema = generator.function_schema(simple_function)
    new_schema = generator.insert_prefix(Reflection, function_schema)
    assert new_schema['name'] == "reflection_and_simple_function"
    assert len(new_schema['parameters']['properties']) == 4
    assert len(new_schema['parameters']['required']) == 3
    assert len(function_schema['parameters']['properties']) == 2  # the old schema is not changed
    assert len(function_schema['parameters']['required']) == 1  # the old schema is not changed
    param_names = list(new_schema['parameters']['properties'].keys())
    assert param_names == ['relevancy', 'next_actions_plan', 'count', 'size']


def test_empty_model_merge():
    class EmptyModel(BaseModel):
        pass

    def function_no_doc(empty_model: EmptyModel):
        pass

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")
        next_actions_plan: str = Field(..., description="What you plan to do next and why")

    generator = SchemaGenerator()

    # Function without docstring and EmptyModelWithDoc as parameter
    function_schema = generator.function_schema(function_no_doc)
    assert function_schema['name'] == 'function_no_doc'
    assert 'parameters' not in function_schema

    new_schema = generator.insert_prefix(Reflection, function_schema)
    assert len(new_schema['parameters']['properties']) == 2
    assert new_schema['name'] == 'reflection_and_function_no_doc'


def test_union_schema():
    class Reflection(BaseModel):
        how_relevant: Union[Literal[1, 2, 3, 4, 5], Literal['1', '2', '3', '4', '5']] = Field(
            ...,
            description="Was the last retrieved information relevant for answering this question? Choose 1, 2, 3, 4, or 5."
        )
        @field_validator('how_relevant')
        @classmethod
        def ensure_int(cls, v):
            if isinstance(v, str) and v in {'1', '2', '3', '4', '5'}:
                return int(v)  # Convert to int
            return v

        why_relevant: str = Field(..., description="Why the retrieved information was relevant?")
        next_actions_plan: str = Field(..., description="")

    generator = SchemaGenerator()

    prefix_schema, _ = generator.get_model_schema(Reflection)
    pprint(prefix_schema)
