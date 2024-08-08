import inspect
from typing import Annotated, Callable, Dict, Any, get_origin, Type
from typing_extensions import TypeGuard

import copy
import pydantic as pd
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from pprint import pprint


class LLMFunction:
    def __init__(self, func, schema=None, name=None, description=None):
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__

        if schema:
            self.schema = schema
            if name or description:
                raise ValueError("Cannot specify name or description when providing a complete schema")
        else:
            self.schema = get_function_schema(func)

            if name:
                self.schema['name'] = name

            if description:
                self.schema['description'] = description

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)



def tool_def(function_schema: dict) -> dict:
    return {
        "type": "function",
        "function": function_schema,
    }

def get_tool_defs(
        functions: list[Callable | LLMFunction],
        case_insensitive: bool = False,
        prefix_class: Type[BaseModel]|None = None,
        prefix_schema_name: bool = True,
        strict: bool = False
        ) -> list[dict]:
    result = []
    for function in functions:
        if isinstance(function, LLMFunction):
            fun_schema = function.schema
        else:
            fun_schema = get_function_schema(function, case_insensitive, strict)

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
    model = pd.create_model(f'{function.__name__}_ParameterModel', **fields)
    model.model_rebuild()
    return model


def _recursive_purge_titles(d: Dict[str, Any]) -> None:
    """Remove a titles from a schema recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == 'title' and "type" in d.keys():
                del d[key]
            else:
                _recursive_purge_titles(d[key])

def get_name(func: Callable | LLMFunction, case_insensitive: bool = False) -> str:
    if isinstance(func, LLMFunction):
        schema_name = func.schema['name']
    else:
        schema_name = func.__name__

    if case_insensitive:
        schema_name = schema_name.lower()
    return schema_name

def get_function_schema(function: Callable | LLMFunction, case_insensitive: bool=False, strict: bool=False) -> dict:
    if isinstance(function, LLMFunction):
        if case_insensitive:
            raise ValueError("Cannot case insensitive for LLMFunction")
        return function.schema

    description = ''
    if hasattr(function, '__doc__') and function.__doc__:
        description = function.__doc__

    schema_name = function.__name__
    if case_insensitive:
        schema_name = schema_name.lower()

    function_schema: dict[str, Any] = {
        'name': schema_name,
        'description': description.strip(),
    }
    model = parameters_basemodel_from_function(function)
    model_json_schema = model.model_json_schema()
    if strict:
        model_json_schema = to_strict_json_schema(model_json_schema)
        function_schema['additionalProperties'] = False
    else:
        _recursive_purge_titles(model_json_schema)
    function_schema['parameters'] = model_json_schema

    return function_schema

# copied from openai implementation which also uses Apache 2.0 license

def to_strict_json_schema(schema: dict) -> dict[str, Any]:
    return _ensure_strict_json_schema(schema, path=())

def _ensure_strict_json_schema(
    json_schema: object,
    path: tuple[str, ...],
) -> dict[str, Any]:
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the API expects.
    """
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = [prop for prop in properties.keys()]
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(prop_schema, path=(*path, "properties", key))
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"))

    # unions
    any_of = json_schema.get("anyOf")
    if isinstance(any_of, list):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i))) for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if isinstance(all_of, list):
        json_schema["allOf"] = [
            _ensure_strict_json_schema(entry, path=(*path, "anyOf", str(i))) for i, entry in enumerate(all_of)
        ]

    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name))

    return json_schema


def is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    # just pretend that we know there are only `str` keys
    # as that check is not worth the performance cost
    return isinstance(obj, dict)

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

    altered_function = LLMFunction(function_with_doc, name="altered_name")

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
        altered_function,
        User
        ]))
