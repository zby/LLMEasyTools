from llm_easy_tools import get_tool_defs, process_response 
from llm_easy_tools.schema_generator import get_name, llm_function

from pydantic import BaseModel
from openai import OpenAI
from pprint import pprint

from groq import Groq
client = Groq()

# with Llama the function name needs to be better fit the action
@llm_function("extract_user_details")
class UserDetail(BaseModel):
    name: str
    city: str

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": "Extract user details from the following sentence: John lives in Warsaw and likes banana"}],
    tools=get_tool_defs([UserDetail]),
    tool_choice="auto"
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [UserDetail])

# #pprint(results)
pprint(results[0].output)
