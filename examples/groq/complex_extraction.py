from llm_easy_tools import get_tool_defs, process_response
from pydantic import BaseModel
from pprint import pprint

from pydantic import BaseModel
from pprint import pprint


class Address(BaseModel):
    street: str
    city: str

class Company(BaseModel):
    name: str
    speciality: str
    address: Address


def print_companies(companies: list[Company]):
    pprint(companies)
    return companies

#pprint(get_tool_defs([print_companies]))

file_path = 'examples/Three_Companies_Story.txt'
with open(file_path, 'r') as file:
    story = file.read()

content = f"{story}\n\nPlease print the information about companies mentioned in the text above."

#client = OpenAI()
from groq import Groq
client = Groq()

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": content}],
    tools=get_tool_defs([print_companies]),
    tool_choice="auto",
)


process_response(response, [print_companies])

# OUTPUT
[
    Company(
        name='Aether Innovations', 
        speciality='sustainable energy solutions', 
        address=Address(street='150 Futura Plaza', city='Metropolis')),
    Company(
        name='Gastronauts',
        speciality='culinary startup',
        address=Address(street='45 Flavor Street', city='Metropolis')),
    Company(
        name='SereneScape',
        speciality='digital wellness',
        address=Address(street='800 Tranquil Trail', city='Metropolis'))]

