from typing import Callable
from pprint import pprint
from typing import Type
from pydantic import BaseModel

from llm_easy_tools.schema_generator import get_function_schema, get_model_schema, get_name, insert_prefix, tool_def, llm_function

class ToolBox:
    def __init__(self,
                 tool_registry=None,
                 fix_json_args=True,
                 case_insensitive=False,
                 insert_prefix_name=True,
                 ):
        self.tool_registry = {} if tool_registry is None else tool_registry
        self.fix_json_args = fix_json_args
        self.case_insensitive = case_insensitive
        self.insert_prefix_name = insert_prefix_name


    def register_function(self, function: Callable):
        function_name = get_name(function, self.case_insensitive)

        if function_name in self.tool_registry:
            raise Exception(f"Trying to register {function_name} which is already registered")

        self.tool_registry[function_name] = { 'function': function }

    def register_model(self, model_class: Type[BaseModel]):
        model_name = get_name(model_class, self.case_insensitive)

        if model_name in self.tool_registry:
            raise Exception(f"Trying to register {model_name} which is already registered")

        self.tool_registry[model_name] = { 'model_class': model_class}

    def tool_schemas(self, prefix_class=None):
        schemas = []
        for tool in self.tool_registry.values():
            if 'function' in tool:
                the_schema = get_function_schema(tool['function'])
            elif 'model_class' in tool:
                the_schema = get_model_schema(tool['model_class'])
            if prefix_class is not None:
                function_schema = insert_prefix(prefix_class, the_schema, self.insert_prefix_name, self.case_insensitive)
            schemas.append(tool_def(the_schema))
        return schemas

#######################################
#
# Examples

@llm_function(schema_name="altered_name")
def function_decorated():
    pass

class ExampleClass:
     def simple_method(self, count: int, size: float):
         """simple method does something"""
         pass

example_object = ExampleClass()

if __name__ == "__main__":
    toolbox = ToolBox()
    toolbox.register_function(function_decorated)
    toolbox.register_function(example_object.simple_method)
#    pprint(generate_tools(function_with_doc, function_decorated))
    pprint(toolbox.tool_schemas())

