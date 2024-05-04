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

- **Schema Generation**: Effortlessly create JSON schemas for tools from type annotations
- **Structured Data from LLM**
- **Function Name Mapping**: Flexibly map JSON schema names to Python code.
- **Dispatching Function Calls**: Directly invoke functions based on LLM response structures.
- **Stateful Tools**: You can register methods bound to an object to have stateful tools. See [examples/stateful_search.py](https://github.com/zby/LLMEasyTools/tree/main/examples)
- **No Patching!**: No globals, some 400 lines of straightforward object oriented code.


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

### Basic Example: Dispatching to a function

```python
from llm_easy_tools import ToolBox
from pydantic import BaseModel
from openai import OpenAI
from pprint import pprint

client = OpenAI()

# Create a ToolBox instance
toolbox = ToolBox()

def contact_user(name: str, city: str):
    return f"User {name} from {city} was contacted"

toolbox.register_function(contact_user)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Contact John. John lives in Warsaw"}],
    tools=toolbox.tool_schemas(),
    tool_choice={"type": "function", "function": {"name": "contact_user"}},
)
# There might be more than one tool calls and more than one result
results = toolbox.process_response(response)

pprint(results[0].output)

```
Output:
```
User John from Warsaw was contacted
```

### Example: Getting structured data from LLM

```python
# Define a Pydantic model for your tool's input
class UserDetail(BaseModel):
    name: str
    city: str

# Register your model
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

pprint(results[0].output)
```
Output:
```
UserDetail(name='John', city='Warsaw')
```


Discover more possibilities and examples in the examples directory and test suite.

## License

LLMEasyTools is open-sourced under the Apache License. For more details, see the LICENSE file.
