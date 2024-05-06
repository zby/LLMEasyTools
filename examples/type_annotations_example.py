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
