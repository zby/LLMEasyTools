from llm_easy_tools import get_tool_defs, process_response
from openai import OpenAI
from pprint import pprint

client = OpenAI()
# This assumes that OpenAI credentials are in the environment variable OPENAI_API_KEY

def get_current_weather(location: str, format: str) -> str:
    """
    Get the current weather for a specified location in either Celsius or Fahrenheit.
    """
    # This is a stub function, so we'll return a placeholder string.
    return f"The current weather in {location} is 20°C with light rain."

# type annotations are crucial here

tool_schemas = get_tool_defs([get_current_weather])

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "user", "content": "What is the weather in Warsaw? Please use Celsius for temperature."}],
    tools=tool_schemas,
    tool_choice="auto",
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [get_current_weather])

for result in results:
    pprint(result.output)

# OUTPUT:
# 'The current weather in Warsaw is 20°C with light rain.'




