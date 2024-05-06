from llm_easy_tools import get_tool_defs, process_response
from pydantic import BaseModel
from openai import OpenAI
from pprint import pprint

client = OpenAI()

# Define a Pydantic model for your tool's output
class UserDetail(BaseModel):
    name: str
    city: str


response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Extract user details from the following sentence: John lives in Warsaw and likes banana"}],
    tools=get_tool_defs([UserDetail]),
    tool_choice="auto",
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [UserDetail])

#pprint(results)
pprint(results[0].output)

