from llm_tool_box import ToolBox
from pydantic import BaseModel
from openai import OpenAI
from pprint import pprint

client = OpenAI()


# Define a Pydantic model for your tool's input
class UserDetail(BaseModel):
    name: str
    city: str


def frobnicate_user(user: UserDetail):
    return f"User {user.name} from {user.city} was frobnicated"

# Create a ToolBox instance
toolbox = ToolBox()

# Register your tool - if a class is passed an identity function over it is registered
toolbox.register_tool(UserDetail)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Extract John lives in Warsaw"}],
    tools=toolbox.tool_schemas,
    tool_choice="auto",
)
# There might be more than one tool calls and more than one result
results = toolbox.process_response(response)

pprint(results)

toolbox.register_tool(frobnicate_user)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Extract John lives in Warsaw"}],
    tools=toolbox.tool_schemas,
    tool_choice={"type": "function", "function": {"name": "frobnicate_user"}},
)
# There might be more than one tool calls and more than one result
results = toolbox.process_response(response)

pprint(results)
