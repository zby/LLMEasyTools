import json

from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice
from pprint import pprint

chat_completion_message = ChatCompletionMessage(
    role="assistant",
    tool_calls=[
        {
            "id": 'A',
            "type": 'function',
            "function": {
                "arguments": json.dumps({"param1": "value1", "param2": "value2"}),
                "name": 'function_name'
            }
        }
    ]
)


pprint(chat_completion_message)

choice = Choice(
    finish_reason="stop",
    index=0,
    message=chat_completion_message
)

response = ChatCompletion(
    id='A',
    created=0,
    model='A',
    choices=[ choice ],
    object='chat.completion'
)

pprint(response)