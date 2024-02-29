import inspect
import json
from typing import Any, Dict, get_type_hints

from docstring_parser import parse
from pydantic import BaseModel


def external_function(schema_name=None):
    def decorator(func):
        setattr(func, 'LLMEasyTools_external_function', True)
        if schema_name is not None:
            setattr(func, 'LLMEasyTools_schema_name', schema_name)
        return func
    return decorator

def extraction_model(schema_name=None):
    def class_decorator(cls):
        setattr(cls, 'LLMEasyTools_extraction_model', True)
        if schema_name is not None:
            setattr(cls, 'LLMEasyTools_schema_name', schema_name)
        return cls
    return class_decorator

class SchemaGenerator:
    def __init__(self, strict=True, name_mappings=None):
        if name_mappings is None:
            name_mappings = []
        self.strict = strict
        self.name_mappings = name_mappings

    def func_name_to_schema(self, func_name):
        for fname, sname in self.name_mappings:
            if func_name == fname:
                return sname
        return func_name

    def schema_name_to_func(self, schema_name):
        for fname, sname in self.name_mappings:
            if schema_name == sname:
                return fname
        return schema_name


    @classmethod
    def _recursive_purge_titles(cls, d: Dict[str, Any]) -> None:
        """Remove a titles from a schema recursively"""
        if isinstance(d, dict):
            for key in list(d.keys()):
                if key == 'title' and "type" in d.keys():
                    del d[key]
                else:
                    cls._recursive_purge_titles(d[key])

    from typing import Callable, Any, Dict

    def function_schema(self, function: Callable) -> Dict[str, Any]:
        description = ''
        parameters = inspect.signature(function).parameters
        if not len(parameters) == 1:
            raise TypeError(f"Function {function.__name__} requires {len(parameters)} parameters but we generate schemas only for one parameter functions")

        # there's exactly one parameter
        name, param = list(parameters.items())[0]
        param_class = param.annotation

        if not issubclass(param_class, BaseModel):
            raise TypeError(f"The only parameter of function {function.__name__} is not a subclass of pydantic BaseModel")

        params_schema = param_class.model_json_schema()
        self._recursive_purge_titles(params_schema)

        if 'description' in params_schema:
            description = params_schema['description']
            params_schema.pop('description')
        if function.__doc__:
            if description and self.strict:
                raise ValueError(f"Both function '{function.__name__}' and the parameter class '{param_class.__name__}' have descriptions")
            else:
                description = parse(function.__doc__).short_description

        schema = {
            "name": self.func_name_to_schema(function.__name__),
            "description": description,
        }
        if len(param_class.__annotations__) > 0:  # if the model is not empty,
            schema["parameters"] = params_schema

        return schema

    def generate_tools(self, *functions: Callable) -> list:
        """
        Generates a tools description array for multiple functions.

        Args:
        *functions: A variable number of functions to introspect.

        Returns:
        A list representing the tools structure for a client.chat.completions.create call.
        """
        tools_array = []
        for function in functions:
            tools_array.append(self.generate_tool_schema(function))
        return tools_array

    def generate_tool_schema(self, function: Callable) -> dict:
        function_schema = self.function_schema(function)
        return {
            "type": "function",
            "function": function_schema,
        }
    def generate_functions(self, *functions: Callable) -> list:
        """
        Generates a functions description array for multiple functions.

        Args:
        *functions: A variable number of functions to introspect.

        Returns:
        A list representing the functions structure for a client.chat.completions.create call.
        """
        functions_array = []
        for function in functions:
            # Check return type
            return_type = get_type_hints(function).get('return')
            if return_type is not None and return_type != str:
                raise ValueError(f"Return type of {function.__name__} is not str")

            functions_array.append(self.function_schema(function))
        return functions_array


class ToolBox:
    """
    A `ToolBox` object can register LLM tools, generate tool schemas that can be loaded into an LLM call,
    and process a function call from an LLM response.

    Methods:
    - register_tool(function): Registers a function as a tool.
    The function need to take exactly one parameter of a class that is a subclass of pydantic BaseModel.
    If the function is a method it needs to be a bound method with one parameter.

    - `toolbox_from_object(cls, obj, *args, **kwargs)`: Creates a new `ToolBox`
        instance from an existing object and register all its public methods
        as LLM tools. Parameters are:
        - `obj`: The object to register tools from.
        - `*args, **kwargs`: Additional parameters passed to `__init__` method.

    - `process(self, function_call)`: Dispatch a function call from an LLM response to the registered function
        that matches the function name.

    Attributes:
    - `name_mappings`: A list of tuples mapping names used in LLM schemas and tool function names used in code.
    - `tool_registry`: A dictionary mapping tool function names to their info
    - `tool_schemas`: A list of tool schemas that can be used in an LLM call.
    """
    def __init__(self, strict=True, name_mappings=None,
                 tool_registry=None, generator=None,
                 tool_schemas=None, tool_sets=None,
                 function_schemas=None,
                 ):
        self.strict = strict
        if tool_registry is None:
            tool_registry = {}
        self.tool_registry = tool_registry
        if name_mappings is None:
            name_mappings = []
        self.name_mappings = name_mappings
        if tool_schemas is None:
            tool_schemas = []
        self.tool_schemas = tool_schemas
        if tool_sets is None:
            tool_sets = {}
        self.tool_sets = tool_sets
        if function_schemas is None:
            function_schemas = []
        self.function_schemas = function_schemas

        if generator is None:
            generator = SchemaGenerator(strict=self.strict, name_mappings=self.name_mappings)
        self.generator = generator

    @classmethod
    def toolbox_from_object(cls, obj, *args, **kwargs):
        instance = cls(*args, **kwargs)
        instance.register_toolset(obj)
        return instance

    def register_toolset(self, obj, key=None):
        if key is None:
            key = type(obj).__name__

        if key in self.tool_sets:
            raise Exception(f"A toolset with key {key} already exists.")

        self.tool_sets[key] = obj
        methods = inspect.getmembers(obj, predicate=inspect.ismethod)
        for name, method in methods:
            if hasattr(method, 'LLMEasyTools_external_function'):
                self.register_function(method)
        for attr_name in dir(obj.__class__):
            attr_value = getattr(obj.__class__, attr_name)
            if isinstance(attr_value, type) and hasattr(attr_value, 'LLMEasyTools_extraction_model'):
                self.register_model(attr_value)

    def register_model(self, model_class):
        if not inspect.isclass(model_class) or not issubclass(model_class, BaseModel):
            raise TypeError("Class must be a Pydantic model class - a subclass of BaseModel")

        def function(obj: model_class) -> model_class:
            return obj
        function.__name__ = model_class.__name__

        if hasattr(model_class, 'LLMEasyTools_schema_name'):
            schema_name = model_class.LLMEasyTools_schema_name
            setattr(function, 'LLMEasyTools_schema_name', schema_name)

        self.register_function(function)


    def register_function(self, function):
        if not inspect.isroutine(function):
            raise TypeError("Parameter must be a function")

        if function.__name__ in self.tool_registry:
            raise Exception(f"Trying to register {function.__name__} which is already registered")

        parameters = inspect.signature(function).parameters
        if len(parameters) != 1:
            raise TypeError(
                f"Function {function.__name__} requires {len(parameters)} parameters but we work only with one parameter functions")

        name, param = list(parameters.items())[0]
        param_class = param.annotation
        if not inspect.isclass(param_class) or not issubclass(param_class, BaseModel):
            raise TypeError(
                f"The only parameter of function {function.__name__} is not a subclass of pydantic BaseModel")

        if hasattr(function, 'LLMEasyTools_schema_name'):
            self.name_mappings.append((function.__name__, function.LLMEasyTools_schema_name))

        tool_schema = self.generator.generate_tool_schema(function)
        self.tool_schemas.append(tool_schema)
        function_schema = self.generator.function_schema(function)
        self.function_schemas.append(function_schema)
        self.tool_registry[function.__name__] = {
            "function": function,
            "param_class": param_class,
            "function_schema": function_schema,
            "tool_schema": tool_schema
        }


    def schema_name_to_func(self, schema_name):
        for fname, sname in self.name_mappings:
            if schema_name == sname:
                return fname
        return schema_name

    def process_response(self, response, choice_num=0):
        results = []
        if response.choices[choice_num].message.function_call:
            function_call = response.choices[choice_num].message.function_call
            results.append(self.process_function(function_call))
        if response.choices[choice_num].message.tool_calls:
            for tool_call in response.choices[choice_num].message.tool_calls:
                results.append(self.process_function(tool_call.function))
        return results

    def process_function(self, function_call):
        tool_args = json.loads(function_call.arguments)
        tool_name = function_call.name
        return self._process_unpacked(tool_name, tool_args)

    def _process_unpacked(self, tool_name, tool_args):
        function_name = self.schema_name_to_func(tool_name)
        if function_name not in self.tool_registry:
            raise ValueError(f"Unknown tool name: {tool_name}")
        function_info = self.tool_registry[function_name]
        param_class = function_info["param_class"]
        function = function_info["function"]
        param = param_class(**tool_args)
        observations = function(param)
        return observations

