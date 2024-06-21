import pytest

from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field, field_validator

from llm_easy_tools import get_function_schema, insert_prefix, llm_function 

from llm_easy_tools.schema_generator import parameters_basemodel_from_function, _recursive_purge_titles, get_name

from pprint import pprint


def simple_function(count: int, size: Optional[float] = None):
    """simple function does something"""
    pass


def simple_function_no_docstring(
        apple: Annotated[str, 'The apple'],
        banana: Annotated[str, 'The banana']
):
    pass




def test_function_schema():

    function_schema = get_function_schema(simple_function)
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

def test_noparams():
    def function_with_no_params():
        """
        This function has a docstring and takes no parameters.
        """
        pass

    def function_no_doc():
        pass

    result = get_function_schema(function_with_no_params)
    assert result['name'] == 'function_with_no_params'
    assert result['description'] == "This function has a docstring and takes no parameters."
    assert result['parameters']['properties'] == {}

    # Function without docstring and EmptyModel as parameter
    result = get_function_schema(function_no_doc)
    assert result['name'] == 'function_no_doc'
    assert result['description'] == ''
    assert result['parameters']['properties'] == {}


def test_nested():
    class Foo(BaseModel):
        count: int
        size: Optional[float] = None

    class Bar(BaseModel):
        """Some Bar"""
        apple: str = Field(description="The apple")
        banana: str = Field(description="The banana")

    class FooAndBar(BaseModel):
        foo: Foo
        bar: Bar

    def nested_structure_function(foo: Foo, bars: List[Bar]):
        """spams everything"""
        pass

    function_schema = get_function_schema(nested_structure_function)
    assert function_schema['name'] == 'nested_structure_function'
    assert function_schema['description'] == 'spams everything'
    assert len(function_schema['parameters']['properties']) == 2

    function_schema = get_function_schema(FooAndBar)
    assert function_schema['name'] == 'FooAndBar'
    assert len(function_schema['parameters']['properties']) == 2


def test_methods():
    class ExampleClass:
        def simple_method(self, count: int, size: Optional[float] = None):
            """simple method does something"""
            pass

    example_object = ExampleClass()

    function_schema = get_function_schema(example_object.simple_method)
    assert function_schema['name'] == 'simple_method'
    assert function_schema['description'] == 'simple method does something'
    params_schema = function_schema['parameters']
    assert len(params_schema['properties']) == 2

def test_name_change():
    @llm_function('changed_name')
    def new_simple_function(count: int, size: Optional[float] = None):
        """simple function does something"""
        pass

    function_schema = get_function_schema(new_simple_function)
    assert function_schema['name'] == 'changed_name'


def test_merge_schemas():

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")
        next_actions_plan: str = Field(..., description="What you plan to do next and why")

    function_schema = get_function_schema(simple_function)
    new_schema = insert_prefix(Reflection, function_schema)
    assert new_schema['name'] == "Reflection_and_simple_function"
    assert len(new_schema['parameters']['properties']) == 4
    assert len(new_schema['parameters']['required']) == 3
    assert len(function_schema['parameters']['properties']) == 2  # the old schema is not changed
    assert len(function_schema['parameters']['required']) == 1  # the old schema is not changed
    param_names = list(new_schema['parameters']['properties'].keys())
    assert param_names == ['relevancy', 'next_actions_plan', 'count', 'size']

    function_schema = get_function_schema(simple_function)
    new_schema = insert_prefix(Reflection, function_schema, case_insensitive=True)
    assert new_schema['name'] == "reflection_and_simple_function"

def test_noparams_function_merge():

    def function_no_params():
        pass

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Whas the last retrieved information relevant and why?")
        next_actions_plan: str = Field(..., description="What you plan to do next and why")

    # Function without docstring and EmptyModelWithDoc as parameter
    function_schema = get_function_schema(function_no_params)
    assert function_schema['name'] == 'function_no_params'
    assert function_schema['parameters']['properties'] == {}
    pprint(function_schema)

    new_schema = insert_prefix(Reflection, function_schema)
    assert len(new_schema['parameters']['properties']) == 2
    assert new_schema['name'] == 'Reflection_and_function_no_params'

def test_model_init_function():

    class User(BaseModel):
        """A user object"""
        name: str
        city: str


    function_schema = get_function_schema(User)
    assert function_schema['name'] == 'User'
    assert function_schema['description'] == 'A user object'
    assert len(function_schema['parameters']['properties']) == 2
    assert len(function_schema['parameters']['required']) == 2


def test_case_insensitivity():

    class User(BaseModel):
        """A user object"""
        name: str
        city: str

    function_schema = get_function_schema(User, case_insensitive=True)
    assert function_schema['name'] == 'user'
    assert get_name(User, case_insensitive=True) == 'user'

def test_function_no_type_annotation():
    def function_with_missing_type(param):
        return f"Value is {param}"

    with pytest.raises(ValueError) as exc_info:
        get_function_schema(function_with_missing_type)
    assert str(exc_info.value) == "Parameter 'param' has no type annotation"
