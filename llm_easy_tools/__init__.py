import inspect
import copy
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
    def __init__(self, strict=True, name_mappings=None, case_insensitive=True):
        if name_mappings is None:
            name_mappings = []
        self.strict = strict
        self.case_insensitive = case_insensitive
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

    def get_model_schema(self, class_ref: BaseModel):
        # type hints seem not enforced in our test case
        if not issubclass(class_ref, BaseModel):
            raise TypeError(
                f"The given class reference is not a subclass of pydantic BaseModel"
            )
        class_schema = class_ref.model_json_schema()
        self._recursive_purge_titles(class_schema)

        description = class_schema.pop('description', '')
        return class_schema, description

    def function_schema(self, function: Callable) -> Dict[str, Any]:
        parameters = inspect.signature(function).parameters
        if not len(parameters) == 1:
            raise TypeError(
                f"Function {function.__name__} requires {len(parameters)} parameters but we generate schemas only for one parameter functions"
            )
        # there's exactly one parameter
        name, param = list(parameters.items())[0]
        param_class = param.annotation
        params_schema, description = self.get_model_schema(param_class)

        if function.__doc__:
            if description and self.strict:
                raise ValueError(
                    f"Both function '{function.__name__}' and the parameter class '{param_class.__name__}' have descriptions"
                )
            else:
                description = parse(function.__doc__).short_description

        schema = {
            "name": self.func_name_to_schema(function.__name__),
            "description": description,
            "parameters": params_schema
        }

        if len(schema['parameters']['properties']) == 0:  # if the parameters list is empty,
            schema.pop('parameters')

        return schema

    def insert_prefix(self, prefix_class, schema, prefix_schema_name=True):
        prefix_schema, _ = self.get_model_schema(prefix_class)
        if 'parameters' in schema:
            prefix_schema['required'].extend(schema['parameters']['required'])
            for key, value in schema['parameters']['properties'].items():
                prefix_schema['properties'][key] = value
        new_schema = copy.copy(schema)  # Create a shallow copy of the schema
        new_schema['parameters'] = prefix_schema
        if len(new_schema['parameters']['properties']) == 0:  # If the parameters list is empty
            new_schema.pop('parameters')
        if prefix_schema_name:
            if self.case_insensitive:
                prefix_name = prefix_class.__name__.lower()
            else:
                prefix_name = prefix_class.__name__
            new_schema['name'] = prefix_name + "_and_" + schema['name']
        return new_schema


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
            tools_array.append(self.tool_schema(self.function_schema(function)))
        return tools_array

    def tool_schema(self, function_schema: dict) -> dict:
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

    - register_model(class): Registers a pydantic model

    - register_toolset(object): Registers all methods marked as 'external_function'.

    - `process(self, function_call)`: Dispatch a function call from an LLM response to the registered function
        that matches the function name.

    - `tool_schemas(self, prefix_class=None)`: A list of tool schemas that can be used in an LLM call.

    - `function_schemas(self, prefix_clas=None)`: A list of function schemas that can be used in an LLM call (for older API versions).

    Attributes:
    - `name_mappings`: A list of tuples mapping names used in LLM schemas and tool function names used in code.
    - `tool_registry`: A dictionary mapping tool function names to their info
    """
    def __init__(self, strict=True, name_mappings=None, case_insensitive=True,
                 tool_registry=None, generator=None,
                 tool_sets=None,
                 ):
        self.strict = strict
        self.case_insensitive = case_insensitive
        if tool_registry is None:
            tool_registry = {}
        self.tool_registry = tool_registry
        if name_mappings is None:
            name_mappings = []
        self.name_mappings = name_mappings
        if tool_sets is None:
            tool_sets = {}
        self.tool_sets = tool_sets

        if generator is None:
            generator = SchemaGenerator(strict=self.strict, case_insensitive=self.case_insensitive, name_mappings=self.name_mappings)
        self.generator = generator


    def tool_schemas(self, prefix_class=None):
        tool_schemas = [self.generator.tool_schema(function_schema) for function_schema in self.function_schemas(prefix_class)]
        return tool_schemas

    def get_tool_schema(self, name, prefix_class=None):
        # todo - decide if we want use internal or external name here
        function_schema = self.tool_registry[name]["function_schema"]
        if prefix_class:
            function_schema = self.generator.insert_prefix(prefix_class, function_schema)
        return self.generator.tool_schema(function_schema)

    def function_schemas(self, prefix_class=None):
        function_schemas = []
        for func_info in self.tool_registry.values():
            schema = func_info["function_schema"]
            if prefix_class:
                schema = self.generator.insert_prefix(prefix_class, schema)
            function_schemas.append(schema)
        return function_schemas


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
        elif self.case_insensitive:
            if not function.__name__.lower() == function.__name__:
                self.name_mappings.append((function.__name__, function.__name__.lower()))

        function_schema = self.generator.function_schema(function)
        tool_schema = self.generator.tool_schema(function_schema)
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
            result = self.process_function(function_call)
            results.append(result)
        if response.choices[choice_num].message.tool_calls:
            for tool_call in response.choices[choice_num].message.tool_calls:
                result = self.process_function(tool_call.function)
                results.append(result)
        return results

    def process_function(self, function_call, prefix_class=None, ignore_prefix=False):
        tool_args = json.loads(function_call.arguments)
        tool_name = function_call.name
        if prefix_class is not None:
            if not ignore_prefix:
                # todo make better API for returning the prefix
                self.prefix = self._extract_prefix_unpacked(tool_args, prefix_class)
            prefix_name = prefix_class.__name__
            if self.case_insensitive:
                prefix_name = prefix_name.lower()
            if not tool_name.startswith(prefix_name) and not ignore_prefix:
                raise ValueError(f"Trying to decode function call with a name '{tool_name}' not matching prefix '{prefix_name}'")
            elif tool_name.startswith(prefix_name):
                tool_name = tool_name[len(prefix_name + '_and_'):]
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

    def _extract_prefix_unpacked(self, tool_args, prefix_class):
        # modifies tool_args
        prefix_args = {}
        for key in list(tool_args.keys()):  # copy keys to list because we modify the dict while iterating over it
            if key in prefix_class.__annotations__:
                prefix_args[key] = tool_args.pop(key)
        prefix = prefix_class(**prefix_args)
        return(prefix)

