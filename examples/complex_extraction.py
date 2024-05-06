from llm_easy_tools import get_tool_defs, process_response
from pydantic import BaseModel
from openai import OpenAI
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
# Load the content into a variable
with open(file_path, 'r') as file:
    story = file.read()

content = f"{story}\n\nPlease print the information about companies mentioned in the text above."

client = OpenAI()

#response = client.chat.completions.create(
    #model="gpt-3.5-turbo-1106",
    #messages=[{"role": "user", "content": content}],
    #tools=get_tool_defs([print_companies]),
    #tool_choice="auto",
#)

##pprint(response)

#process_response(response, [print_companies])

## OUTPUT

[
    {
        'address': { 'city': 'Metropolis', 'street': '150 Futura Plaza' },
        'name': 'Aether Innovations',
        'speciality': 'sustainable energy solutions'
        }, 
    {
        'address': {'city': 'Metropolis', 'street': '45 Flavor Street'},
        'name': 'Gastronauts',
        'speciality': 'culinary startup'
    },
    {
        'address': {'city': 'Metropolis', 'street': '800 Tranquil Trail'},
        'name': 'SereneScape',
        'speciality': 'digital wellness'
    }
]

# The processor does not create the objects - it leaves them as dictionaries.

class CompanyList(BaseModel):
    companies: list[Company]


response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": content}],
    tools=get_tool_defs([CompanyList]),
    tool_choice="auto",
)

#pprint(response)

results = process_response(response, [CompanyList])
pprint(results[0].output)

# OUTPUT
CompanyList(companies=[Company(name='Aether Innovations', speciality='sustainable energy solutions', address=Address(street='150 Futura Plaza', city='Metropolis')), Company(name='Gastronauts', speciality='urban dining experiences', address=Address(street='45 Flavor Street', city='Metropolis')), Company(name='SereneScape', speciality='digital wellness', address=Address(street='800 Tranquil Trail', city='Metropolis'))])
