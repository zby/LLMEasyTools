from __future__ import annotations

import pytest
from pydantic import BaseModel
from llm_easy_tools.schema_generator import parameters_basemodel_from_function

class Query(BaseModel):
    query: str
    region: str

def test_pydantic_param():
    def search(query: Query):
        ...

    model = parameters_basemodel_from_function(search)
    model_json_schema = model.model_json_schema()
    assert 'query' in model_json_schema['properties']


@pytest.mark.xfail(reason="Local class not currently supported, needs investigation")
def test_pydantic_param_with_local_class():
    class User(BaseModel):
        name: str

    def search(user: User):
        ...

    model = parameters_basemodel_from_function(search)
    model_json_schema = model.model_json_schema()
    assert 'user' in model_json_schema['properties']