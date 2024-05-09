from pydantic import BaseModel
from datetime import date
import json  # Import json module
from pprint import pprint
from llm_easy_tools.schema_generator import get_tool_defs
from llm_easy_tools.processor import process_response
from openai import OpenAI

class User(BaseModel):
    name: str
    birth_date: date

# Creating an object of the User class
#user = User(name="John Doe", birth_date=date(1990, 5, 15))

# Using the .model_dump() method and printing the output
#print(user.model_dump())

# Printing out the JSON schema of the User model using the recommended method
#print(json.dumps(User.model_json_schema(), indent=2))

def print_users(users: list[User]):
    pprint(users)
    return users

#pprint(get_tool_defs([print_companies]))

story = "John Doe was born on 15th May 1990, Mary was his daughter born on 16th May 1991"

content = f"{story}\n\nPlease print the information about users mentioned in the text above."

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": content}],
    tools=get_tool_defs([print_users]),
    tool_choice="auto",
)


process_response(response, [print_users])

