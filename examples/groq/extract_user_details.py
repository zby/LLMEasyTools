from llm_easy_tools import get_tool_defs, process_response
from pydantic import BaseModel
from pprint import pprint

from groq import Groq
client = Groq()


# Define a Pydantic model for your tool's output
class UserDetail(BaseModel):
    name: str
    city: str


response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": "Extract user details from the following sentence: John lives in Warsaw and likes banana"}],
    tools=get_tool_defs([UserDetail]),
    tool_choice={'type': 'function', 'function': {'name': 'UserDetail'}},
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [UserDetail])

# #pprint(results)
pprint(results[0].output)
