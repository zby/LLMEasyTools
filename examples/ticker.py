import yfinance as yf


from llm_easy_tools import get_tool_defs, process_response
from openai import OpenAI

client = OpenAI()
# This assumes that OpenAI credentials are in the environment variable OPENAI_API_KEY

def get_nasdaq_price(ticker: str):
    """Get the current price of a NASDAQ ticker"""
    ticker = yf.Ticker(ticker)
    current_price = ticker.history(period="1d")['Close'].iloc[0]
    return current_price

# type annotations are crucial here

tool_schemas = get_tool_defs([get_nasdaq_price])

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "user", "content": "What is the price of Apple and Microsoft?"}],
    tools=tool_schemas,
    tool_choice="auto",
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [get_nasdaq_price])

for result in results:
    print(f"The price of {result.arguments['ticker']} is {result.output}")



