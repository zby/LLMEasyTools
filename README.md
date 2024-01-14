# LLMToolBox
**OpenAI tools and functions with no fuss.**

LLMToolBox is a minimal Python package designed for seamless interaction with 
[OpenAI Python API library](https://github.com/openai/openai-python).
It focuses on 'tools' and 'function calls', offering a minimalist approach that doesn't get in your way.

One of the key advantages of LLMToolBox is its non-intrusive design. 
It doesn't take over the interaction with the OpenAI client, making it easier for developers
to debug and see exactly what values are being passed.

By integrating Pydantic, LLMToolBox ensures robust data validation and schema generation.

## Features

- **Schema Generation**: Effortlessly create JSON schemas for tools using Pydantic models.
- **Function Name Mapping**: Flexibly map JSON schema names to Python code.
- **Dispatching Function Calls**: Directly invoke functions based on LLM response structures.

## Installation

Install LLMToolBox with these simple steps:

```bash
git clone git@github.com:zby/LLMToolBox.git
cd LLMToolBox
pip install .
```
For development, consider an editable installation:

```bash

pip install -r requirements.txt
pip install -e .
```

Run tests to ensure everything is set up correctly:

```bash
pytest -v tests
```

## Usage

### Basic Example: Data Extraction

```python
from llm_tool_box import ToolBox
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
toolbox.register_tool(UserDetail)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Extract John lives in Warsaw"}],
    tools=toolbox.tool_schemas,
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

### Example: Function Call Processing

```python
def frobnicate_user(user: UserDetail):
    return f"User {user.name} from {user.city} was frobnicated"

toolbox.register_tool(frobnicate_user)

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": "Extract John lives in Warsaw"}],
    tools=toolbox.tool_schemas,
    tool_choice={"type": "function", "function": {"name": "frobnicate_user"}},
)
# There might be more than one tool calls and more than one result
results = toolbox.process_response(response)

pprint(results)
```
Output:
```
['User John from Warsaw was frobnicated']
```

Discover more possibilities and examples in the test suite.

## License

LLMToolBox is open-sourced under the Apache License. For more details, see the LICENSE file.