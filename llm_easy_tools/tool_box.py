import json
from typing import Callable
from pprint import pprint
from typing import Type, Optional, List, Union
from pydantic import BaseModel, ValidationError

from openai.types.chat.chat_completion import ChatCompletionMessage, ChatCompletion, Choice
from openai.types.chat.chat_completion_message_tool_call   import ChatCompletionMessageToolCall, Function

from llm_easy_tools.schema_generator import get_function_schema, get_name, insert_prefix, tool_def, llm_function
from llm_easy_tools.processor import ToolResult, process_response as processor_process_response, get_toolset_tools

class ToolBox:
    def __init__(self,
                 fix_json_args: bool = True,
                 case_insensitive: bool = False,
                 insert_prefix_name: bool = True,
                 ):
        self._tool_registry: dict[str, dict[str, Callable | Type[BaseModel]]] = {}
        self._tool_sets: dict[str, object] = {}
        self.fix_json_args = fix_json_args
        self.case_insensitive = case_insensitive
        self.insert_prefix_name = insert_prefix_name


    def register_function(self, function: Callable) -> None:
        function_name = get_name(function, self.case_insensitive)

        if function_name in self._tool_registry:
            raise Exception(f"Trying to register {function_name} which is already registered")

        self._tool_registry[function_name] = { 'function': function }

    def register_toolset(self, obj: object, key=None) -> None:
        if key is None:
            key = type(obj).__name__

        if key in self._tool_sets:
            raise Exception(f"A toolset with key {key} already exists.")

        self._tool_sets[key] = obj
        for tool in get_toolset_tools(obj):
            self.register_function(tool)


    def get_toolset(self, key: str) -> object:
        return self._tool_sets[key]

    def tool_schemas(self, prefix_class=None, predicate=None) -> list[dict]:
        if predicate is None:
            predicate = lambda _: True
        schemas = []
        for tool_name, tool_value in self._tool_registry.items():
            if not predicate(tool_value):
                continue
            schemas.append(self.get_tool_schema(tool_name, prefix_class))
        return schemas


    def get_tool_schema(self, tool_name: str, prefix_class=None) -> dict:
        tool = self._tool_registry[tool_name]
        the_schema = get_function_schema(tool['function'])
        if prefix_class is not None:
            the_schema = insert_prefix(prefix_class, the_schema, self.insert_prefix_name, self.case_insensitive)
        return tool_def(the_schema)

    def process_response(self, response: ChatCompletion, choice_num=0, prefix_class=None) -> list[ToolResult]:
        functions = [f['function'] for f in self._tool_registry.values()]
        return  processor_process_response(response, functions, choice_num, prefix_class)
    


#######################################
#
# Examples


if __name__ == "__main__":
    @llm_function(schema_name="altered_name")
    def function_decorated():
        return 'Result of function_decorated'

    class ExampleClass:
        def simple_method(self, count: int, size: float):
            """simple method does something"""
            return 'Result of simple_method'

    example_object = ExampleClass()

    toolbox = ToolBox()
    toolbox.register_function(function_decorated)
    toolbox.register_function(example_object.simple_method)
#    pprint(generate_tools(function_with_doc, function_decorated))
    pprint(toolbox.tool_schemas())

    chat_completion_message = ChatCompletionMessage(
        role="assistant",
        tool_calls=[
            ChatCompletionMessageToolCall( 
                id ='A',
                type = 'function',
                function = Function(
                    arguments = json.dumps({"count": 1, "size": 2.2}),
                    name = 'simple_method'
                )
            )
        ]
    )

    chat_completion = ChatCompletion(
        id='A',
        created=0,
        model='A',
        choices=[Choice(finish_reason='stop', index=0, message=chat_completion_message)],
        object='chat.completion'
    )

    print(toolbox.process_response(chat_completion))



