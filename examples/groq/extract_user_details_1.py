from llm_easy_tools import get_tool_defs, process_response, LLMFunction

from pydantic import BaseModel
from pprint import pprint

from groq import Groq
client = Groq()

class UserDetail(BaseModel):
    name: str
    city: str

# with Llama the function name needs to be better fit the action
extract_fun = LLMFunction(
    UserDetail,
    name="extract_user_details",
)

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": "Extract user details from the following sentence: John lives in Warsaw and likes banana"}],
    tools=get_tool_defs([extract_fun]),
    tool_choice="auto"
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [extract_fun])

# #pprint(results)
pprint(results[0].output)
