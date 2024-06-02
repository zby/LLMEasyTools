# LLMEasyTools
**OpenAI tools and functions calls with simplicity**

LLMEasyTool is a minimal Python package designed for seamless interaction with 
[OpenAI Python API library](https://github.com/openai/openai-python) or compatible.
It focuses on *agentic* workflows with the LLM choosing the *tool* appropriate for a task,
offering a minimalist approach that doesn't get in your way.

One of the key advantages of LLMEasyTools is its non-intrusive design. 
It doesn't take over the interaction with the OpenAI client, making it easier for developers
to debug their code and optimize the communication with the LLM.

By integrating Pydantic, LLMEasyTools ensures robust data validation and schema generation.

## Features

- **Schema Generation**: Effortlessly create JSON schemas from type annotations.
- Structured Data from LLM: Works with Pydantic models and basic data structures.
- Function Name Mapping: Flexibly map JSON schema names to Python code. Optional case insensitivity
- Dispatching Function Calls: Directly invoke functions based on LLM response.
- Stateful Tools: You can pass methods bound to an object to have stateful tools. See [examples/stateful_search.py](https://github.com/zby/LLMEasyTools/tree/main/examples).
- Stateless api: Schema generation and dispatching function calls are pure functions, even though tools themselves can be stateful.
- No Patching!: No globals, some 400 lines of mostly straightforward code.
- Parallel tool execution: tool calls from one response are executed in parallel.


## Installation

```bash
pip install LLMEasyTools
```

For development:
```bash
git clone git@github.com:zby/LLMEasyTools.git
cd LLMEasyTools
pip install -e .
pytest -v tests
```

## Usage

There are two parts of the so called *LLM function calling* that are not covered by the standard openai library:
- schema generation
- function dispatching

LLMEasyTools provides two functions to handle this:
- `get_tool_defs` - generates a list of tool definitions (schemas) from a list of functions
- `process_response` - processes a response from the LLM and returns a list of results

**Type annotations are required for all function parameters**

### Basic Example: Dispatching to a function

```python
from llm_easy_tools import get_tool_defs, process_response
from openai import OpenAI
from pprint import pprint

client = OpenAI()


def contact_user(name: str, city: str) -> str:
    return f"User {name} from {city} was contactd"

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Contact John. John lives in Warsaw"}],
    tools=get_tool_defs([contact_user]),
    tool_choice={"type": "function", "function": {"name": "contact_user"}},
)
# There might be more than one tool calls in a single response so results are a list
results = process_response(response, [contact_user])

pprint(results[0].output)
```
Output:
```
User John from Warsaw was contacted
```

### Example: Getting structured data from LLM

```python
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
```
Output:
```
UserDetail(name='John', city='Warsaw')
```

### Example of using type annotations for descriptions in schema:
```python

from llm_easy_tools import get_tool_defs
from typing import Annotated
from pprint import pprint

def contact_user(
        name: Annotated[str, "The name of the user"],
        email: Annotated[str, "The email of the user"],
        phone: Annotated[str, "The phone number of the user"]
        ) -> str:
    """
    Contact the user with the given name, email, and phone number.
    """
    pass


pprint(get_tool_defs([contact_user]))

#  OUPUT

[{
    'function': {
        'description': 'Contact the user with the given name, email, and phone number.',
        'name': 'contact_user',
        'parameters': {
            'properties': {
                'email': {
                    'description': 'The email of the user',
                    'type': 'string'
                },
                'name': {
                    'description': 'The name of the user',
                    'type': 'string'
                },
                'phone': {
                    'description': 'The phone number of the user',
                    'type': 'string'
                }
            },
            'required': ['name', 'email', 'phone'],
            'type': 'object'
        },
        'type': 'function'
    }
}]
```

Discover more possibilities and examples in the examples directory and test suite.

## Limitations
We internally construct a Pydantic model from the function arguments to generate the schema
and then we use that same model to create the function parameters from the LLM response.
So the functions arguments must have [types that can be used in fields in pydantic objects](https://docs.pydantic.dev/latest/concepts/types/#pydantic-types).

For example you cannot get tool definitions for functions that as argument take another function.
This results in an error:
```
def apply_function(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

pprint(get_tool_defs([apply_function]))
```
Another example of unacceptable function signature is a function that takes a socket as an argument.
This produces an error too:
```
def send_message(sock: socket.socket, message: str):
    """
    Sends a message through a socket connection.
    """
    # Ensure the socket is connected
    if sock:
        try:
            # Send the message
            sock.sendall(message.encode('utf-8'))
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Socket is not connected.")

pprint(get_tool_defs([send_message]))
```

## License

LLMEasyTools is open-sourced under the Apache License. For more details, see the LICENSE file.
