import json
import inspect
import traceback

from typing import Callable
from pprint import pprint
from typing import Type, Optional
from pydantic import BaseModel

from llm_easy_tools.schema_generator import get_function_schema, get_model_schema, get_name, insert_prefix, tool_def, llm_function
from openai.types.chat.chat_completion import ChatCompletionMessage, ChatCompletion

class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    output: Optional[str] = None
    model: Optional[BaseModel] = None
    error: Optional[str] = None

class ToolBox:
    def __init__(self,
                 tool_registry=None,
                 fix_json_args=True,
                 case_insensitive=False,
                 insert_prefix_name=True,
                 tool_sets=None
                 ):
        self.tool_registry = {} if tool_registry is None else tool_registry
        self.fix_json_args = fix_json_args
        self.case_insensitive = case_insensitive
        self.insert_prefix_name = insert_prefix_name
        self.tool_sets = {} if tool_sets is None else tool_sets


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
            if isinstance(attr_value, type) and hasattr(attr_value, 'LLMEasyTools_external_function'):
                self.register_model(attr_value)

    def tool_schemas(self, prefix_class=None):
        schemas = []
        for tool_name in self.tool_registry.keys():
            schemas.append(tool_def(self.get_tool_schema(tool_name, prefix_class)))
        return schemas

    def get_tool_schema(self, tool_name, prefix_class=None):
        tool = self.tool_registry[tool_name]
        if 'function' in tool:
            the_schema = get_function_schema(tool['function'])
        elif 'model_class' in tool:
            the_schema = get_model_schema(tool['model_class'])
        if prefix_class is not None:
            the_schema = insert_prefix(prefix_class, the_schema, self.insert_prefix_name, self.case_insensitive)
        return the_schema

    def process_response(self, response, choice_num=0, prefix_class=None, ignore_prefix=False):
        results = []
        if response.choices[choice_num].message.function_call:
            # this is obsolete in openai - but maybe it is used by other llms?
            function_call = response.choices[choice_num].message.function_call
            result = self.process_function(function_call, None, prefix_class, ignore_prefix)
            results.append(result)
        if response.choices[choice_num].message.tool_calls:
            for tool_call in response.choices[choice_num].message.tool_calls:
                result = self.process_function(tool_call.function, tool_call.id, prefix_class, ignore_prefix)
                results.append(result)
        return results

    def process_function(self, function_call, tool_id, prefix_class=None, ignore_prefix=False):
        tool_name = function_call.name
        args = function_call.arguments
        try:
            tool_args = json.loads(args)
        except json.decoder.JSONDecodeError as e:
            if self.fix_json_args:
                args = args.replace(', }', '}').replace(',}', '}')
                tool_args = json.loads(args)
            else:
                error = traceback.format_exc()
                return ToolResult(tool_call_id=tool_id, name=tool_name, error=error)

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
        return self._process_unpacked(tool_name, tool_id, tool_args)


    def _process_unpacked(self, tool_name, tool_id, tool_args=None):
        tool_args = {} if tool_args is None else tool_args
        tool_info = self.tool_registry[tool_name]
        error = None
        if 'model_class' in tool_info:
            model_class = tool_info['model_class']
            model = None
            try:
                model = model_class(**tool_args)
            except Exception as e:
                error = traceback.format_exc()
            result = ToolResult(tool_call_id=tool_id, name=tool_name, model=model, error=error)
        else:
            function = tool_info["function"]
            try:
                output = function(**tool_args)
            except Exception as e:
                error = traceback.format_exc()
            result = ToolResult(tool_call_id=tool_id, name=tool_name, output=output, error=error)

        return result

    def _extract_prefix_unpacked(self, tool_args, prefix_class):
        # modifies tool_args
        prefix_args = {}
        for key in list(tool_args.keys()):  # copy keys to list because we modify the dict while iterating over it
            if key in prefix_class.__annotations__:
                prefix_args[key] = tool_args.pop(key)
        prefix = prefix_class(**prefix_args)
        return(prefix)


#######################################
#
# Examples

@llm_function(schema_name="altered_name")
def function_decorated():
    return 'Result of function_decorated'

class ExampleClass:
     def simple_method(self, count: int, size: float):
         """simple method does something"""
         return 'Result of simple_method'

example_object = ExampleClass()


if __name__ == "__main__":
    toolbox = ToolBox()
    toolbox.register_function(function_decorated)
    toolbox.register_function(example_object.simple_method)
#    pprint(generate_tools(function_with_doc, function_decorated))
    pprint(toolbox.tool_schemas())
    pprint(toolbox._process_unpacked('altered_name', 'aaa'))

    chat_completion_message = ChatCompletionMessage(
        role="assistant",
        tool_calls=[
            {
                "id": 'A',
                "type": 'function',
                "function": {
                    "arguments": json.dumps({"count": 1, "size": 2.2}),
                    "name": 'simple_method'
                }
            }
        ]
    )

    chat_completion = ChatCompletion(
        id='A',
        created=0,
        model='A',
        choices=[{'finish_reason': 'stop', 'index': 0, 'message': chat_completion_message}],
        object='chat.completion'
    )

    print(toolbox.process_response(chat_completion))



