# LLMEasyTools
**OpenAI tools and functions with no fuss.**

LLMEasyTool is a minimal Python package designed for seamless interaction with 
[OpenAI Python API library](https://github.com/openai/openai-python).
It focuses on 'tools' and 'function calls', offering a minimalist approach that doesn't get in your way.

One of the key advantages of LLMEasyTools is its non-intrusive design. 
It doesn't take over the interaction with the OpenAI client, making it easier for developers
to debug their code and optimize the communication with the LLM.

By integrating Pydantic, LLMEasyTools ensures robust data validation and schema generation.

## Features

- **Schema Generation**: Effortlessly create JSON schemas for tools using Pydantic models.
- **Structured Data from LLM**
- **Function Name Mapping**: Flexibly map JSON schema names to Python code.
- **Dispatching Function Calls**: Directly invoke functions based on LLM response structures.
- **Stateful Tools**: You can register methods bound to an object to have stateful tools. See [examples/stateful_search.py](https://github.com/zby/LLMEasyTools/tree/main/examples)
- **No Patching!**: No globals, some 200 lines of straightforward object oriented code.

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

### Basic Example: Getting structured data from LLM

```python
from llm_easy_tools import ToolBox
from pydantic import BaseModel
from openai import OpenAI
from pprint import pprint

client = OpenAI()


# Define a Pydantic model for your tool's input
class UserDetail(BaseModel):
    name: str
    city: str


# Create a ToolBox instance
toolbox = ToolBox()

# Register your tool - if a class is passed an identity function over it is registered
toolbox.register_model(UserDetail)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user",
               "content": "Extract user details from the following sentence: John lives in Warsaw and likes banana"}],
    tools=toolbox.tool_schemas(),
    tool_choice="auto",
)

# There might be more than one tool calls and more than one result
results = toolbox.process_response(response)

pprint(results)
```
Output:
```
[UserDetail(name='John', city='Warsaw')]
```

### Example: Dispatching to a function

```python
def contact_user(user: UserDetail):
    return f"User {user.name} from {user.city} was contacted"


toolbox.register_model(contact_user)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Contact John. John lives in Warsaw"}],
    tools=toolbox.tool_schemas(),
    tool_choice={"type": "function", "function": {"name": "contact_user"}},
)
# There might be more than one tool calls and more than one result
results = toolbox.process_response(response)

pprint(results)

```
Output:
```
['User John from Warsaw was contacted']
```

Discover more possibilities and examples in the examples directory and test suite.

## License

LLMEasyTools is open-sourced under the Apache License. For more details, see the LICENSE file.
