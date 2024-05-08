from llm_easy_tools import get_tool_defs, process_response
from pprint import pprint
from groq import Groq
client = Groq()


def contact_user(name: str, city: str) -> str:
    return f"User {name} from {city} was contactd"


response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": "Contact John. John lives in Warsaw"}],
    tools=get_tool_defs([contact_user]),
    tool_choice={"type": "function", "function": {"name": "contact_user"}},
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [contact_user])

pprint(results[0].output)
