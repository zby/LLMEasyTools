from typing import Protocol, Optional, runtime_checkable
from dataclasses import dataclass

import json

@runtime_checkable
class Function(Protocol):
    name: str
    arguments: str

@runtime_checkable
class ChatCompletionMessageToolCall(Protocol):
    id: str
    function: Function
    type: str

@runtime_checkable
class ChatCompletionMessage(Protocol):
    role: str
    tool_calls: Optional[list[ChatCompletionMessageToolCall]]
    function_call: Optional[Function]

@runtime_checkable
class ChatCompletionChoice(Protocol):
    finish_reason: str
    index: int
    message: ChatCompletionMessage

@runtime_checkable
class ChatCompletion(Protocol):
    id: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    object: str
    
# for testing we need concrete types instead of the Protocols we have above
@dataclass
class SimpleFunction:
    name: str
    arguments: str

@dataclass
class SimpleToolCall:
    id: str
    function: SimpleFunction
    type: str = 'function'

@dataclass
class SimpleMessage:
    role: str
    tool_calls: Optional[list[SimpleToolCall]] = None
    function_call: Optional[SimpleFunction] = None

@dataclass
class SimpleChoice:
    finish_reason: str
    index: int
    message: SimpleMessage

@dataclass
class SimpleCompletion:
    id: str
    created: int
    model: str
    choices: list[SimpleChoice]
    object: str = 'chat.completion'

def mk_chat_with_tool_call(name, args):
    message = SimpleMessage(
        role="assistant",
        tool_calls=[
            SimpleToolCall(
                id='A',
                function=SimpleFunction(
                    arguments=json.dumps(args),
                    name=name
                )
            )
        ]
    )
    chat_completion = SimpleCompletion(
        id='A',
        created=0,
        model='A',
        choices=[SimpleChoice(finish_reason='stop', index=0, message=message)],
        object='chat.completion'
    )
    return chat_completion


