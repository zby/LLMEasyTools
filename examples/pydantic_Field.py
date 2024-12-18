from llm_easy_tools import get_tool_defs
from pydantic import Field
from pprint import pprint

def contact_user(
        name: str = Field(description="The name of the user"),
        email: str = Field(description="The email of the user"),
        phone: str = Field(description="The phone number of the user")
        ) -> str:
    """
    Contact the user with the given name, email, and phone number.
    """
    pass

pprint(get_tool_defs([contact_user]))