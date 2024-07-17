from llm_easy_tools import get_tool_defs, process_response
from openai import OpenAI
from pprint import pprint

client = OpenAI()
# This assumes that OpenAI credentials are in the environment variable OPENAI_API_KEY

def contact_user(name: str, city: str) -> str:
    return f"User {name} from {city} was contacted"

# type annotations are crucial here

tool_schemas = get_tool_defs([contact_user])

text = "John lives in Warsaw. Bob lives in London."

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": "You are a personal assistant. Your current task is to contact all users mentioned in the message."},
        {"role": "user", "content": text}],
    tools=tool_schemas,
    tool_choice="auto",
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [contact_user])

for result in results:
    pprint(result.output)


# OUTPUT:
# 'User John from Warsaw was contacted'
# 'User Bob from London was contacted'