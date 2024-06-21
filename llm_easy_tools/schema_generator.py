import inspect
from typing import Annotated, Callable, Dict, Any, get_origin, Type
from openai.types.chat import ChatCompletionToolParam

import copy
import pydantic as pd
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from pprint import pprint


def llm_function(schema_name=None):
    def decorator(func):
        setattr(func, 'LLMEasyTools_external_function', True)
        if schema_name is not None:
            setattr(func, 'LLMEasyTools_schema_name', schema_name)
        return func
    return decorator



def tool_def(function_schema: dict) -> dict:
    return {
        "type": "function",
        "function": function_schema,
    }

def get_tool_defs(
        functions: list[Callable],
        case_insensitive: bool = False,
        prefix_class: Type[BaseModel]|None = None,
        prefix_schema_name: bool = True
        ) -> list[ChatCompletionToolParam]:
    result = []
    for function in functions:
        fun_schema = get_function_schema(function, case_insensitive)
        if prefix_class:
            fun_schema = insert_prefix(prefix_class, fun_schema, prefix_schema_name, case_insensitive)
        result.append(tool_def(fun_schema))
    return result

def parameters_basemodel_from_function(function: Callable) -> Type[pd.BaseModel]:
    fields = {}
    parameters = inspect.signature(function).parameters
    for name, parameter in parameters.items():
        description = None
        type_ = parameter.annotation
        if type_ is inspect._empty:
            raise ValueError(f"Parameter '{name}' has no type annotation")
        if get_origin(parameter.annotation) is Annotated:
            if parameter.annotation.__metadata__:
                description = parameter.annotation.__metadata__[0]
            type_ = parameter.annotation.__args__[0]
        default = PydanticUndefined if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (type_, pd.Field(default, description=description))
    return pd.create_model(f'{function.__name__}_ParameterModel', **fields)


def _recursive_purge_titles(d: Dict[str, Any]) -> None:
    """Remove a titles from a schema recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == 'title' and "type" in d.keys():
                del d[key]
            else:
                _recursive_purge_titles(d[key])

def get_name(func: Callable, case_insensitive: bool = False) -> str:
    schema_name = func.__name__
    if hasattr(func, 'LLMEasyTools_schema_name'):
        schema_name = func.LLMEasyTools_schema_name
    if case_insensitive:
        schema_name = schema_name.lower()
    return schema_name


def get_function_schema(function: Callable, case_insensitive: bool=False) -> dict:
    description = ''
    if hasattr(function, 'LLMEasyTools_description'):
        description = function.LLMEasyTools_description
    elif hasattr(function, '__doc__') and function.__doc__:
        description = function.__doc__
    function_schema: dict[str, Any] = {
        'name': get_name(function, case_insensitive),
        'description': description.strip(),
    }
    model = parameters_basemodel_from_function(function)
    model_json_schema = model.model_json_schema()
    _recursive_purge_titles(model_json_schema)
    function_schema['parameters'] = model_json_schema
    return function_schema

def insert_prefix(prefix_class, schema, prefix_schema_name=True, case_insensitive = False):
    if not issubclass(prefix_class, BaseModel):
        raise TypeError(
            f"The given class reference is not a subclass of pydantic BaseModel"
        )
    prefix_schema = prefix_class.model_json_schema()
    _recursive_purge_titles(prefix_schema)
    prefix_schema.pop('description', '')

    if 'parameters' in schema:
        required = schema['parameters'].get('required', [])
        prefix_schema['required'].extend(required)
        for key, value in schema['parameters']['properties'].items():
            prefix_schema['properties'][key] = value
    new_schema = copy.copy(schema)  # Create a shallow copy of the schema
    new_schema['parameters'] = prefix_schema
    if len(new_schema['parameters']['properties']) == 0:  # If the parameters list is empty
        new_schema.pop('parameters')
    if prefix_schema_name:
        if case_insensitive:
            prefix_name = prefix_class.__name__.lower()
        else:
            prefix_name = prefix_class.__name__
        new_schema['name'] = prefix_name + "_and_" + schema['name']
    return new_schema


#######################################
#
# Examples

if __name__ == "__main__":
    def function_with_doc():
        """
        This function has a docstring and no parameteres.
        Expected Cost: high
        """
        pass

    @llm_function(schema_name="altered_name")
    def function_decorated():
        pass

    class ExampleClass:
        def simple_method(self, count: int, size: float):
            """simple method does something"""
            pass

    example_object = ExampleClass()

    class User(BaseModel):
        name: str
        age: int

    pprint(get_tool_defs([
        example_object.simple_method, 
        function_with_doc, 
        function_decorated,
        User
        ]))

