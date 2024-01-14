from llm_tool_box import ToolBox, ToolResult
from pydantic import BaseModel
from openai import OpenAI
from pprint import pprint

client = OpenAI()


# Define a Pydantic model for your tool's input
class UserDetail(BaseModel):
    name: str
    age: int


# Create a ToolBox instance
toolbox = ToolBox()

# Register your tool - if a class is passed an identity function over it is registered
toolbox.register_tool(UserDetail)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Extract Jason is 25 years old"}],
    tools=toolbox.tool_schemas,
    tool_choice="auto",
)
pprint(response.choices[0])
response_message = response.choices[0].message
tool_calls = response_message.tool_calls
for tool_call in tool_calls:
    function_call = tool_call.function
    result = toolbox.process(function_call)

pprint(result)
