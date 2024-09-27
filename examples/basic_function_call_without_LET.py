# This example shows how to use function calling without using the llm_easy_tools library

from openai import OpenAI
from pprint import pprint
import json

client = OpenAI()
# This assumes that OpenAI credentials are in the environment variable OPENAI_API_KEY

def contact_user(name: str, city: str) -> str:
    return f"User {name} from {city} was contacted"

# type annotations are crucial here

tool_schemas = [
    {
        "function": {
            "description": "",
            "name": "contact_user",
            "parameters": {
                "properties": {
                    "city": {"type": "string"},
                    "name": {"type": "string"}
                },
                "required": ["name", "city"],
                "type": "object",
            },
        },
        "type": "function",
    }
]

text = "John lives in Warsaw. Bob lives in London."

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": "You are a personal assistant. Your current task is to contact all users mentioned in the message."},
        {"role": "user", "content": text}],
    tools=tool_schemas,
    tool_choice="auto",
)

# you need to loop over all tool calls and process them one by one
for tool_call in response.choices[0].message.tool_calls:
    # you need to get the function object from the global namespace
    function = globals()[tool_call.function.name]
    # and then parse the arguments from the tool call
    function_args = json.loads(tool_call.function.arguments)
    # and then call it with the arguments from the tool call
    result = function(**function_args)
    print(f"Result of function call: {result}")


# OUTPUT:
# 'User John from Warsaw was contacted'
# 'User Bob from London was contacted'
