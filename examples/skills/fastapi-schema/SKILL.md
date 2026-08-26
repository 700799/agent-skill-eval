---
name: fastapi-schema
description: Use this skill when defining or refactoring FastAPI request/response payloads, or whenever a route returns a raw dict. Enforces Pydantic v2 models with field_validator and explicit response_model.
---

# FastAPI schema conventions

Every route must return a typed Pydantic **v2** model — never a raw `dict`.

## Rules

1. Declare a `BaseModel` subclass per payload shape.
2. Annotate every field; no bare `Any`.
3. Validate with `@field_validator` + `@classmethod`. The v1 `@validator`
   decorator is forbidden — it is removed in Pydantic v2.
4. Pass the model to the route via `response_model=`.

## Example

```python
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

app = FastAPI()


class Item(BaseModel):
    id: int
    name: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("name must not be empty")
        return value


@app.get("/items", response_model=Item)
def read_items() -> Item:
    return Item(id=1, name="Item")
```

## Anti-patterns

```python
# WRONG: v1 syntax, removed in Pydantic v2
@validator("name")
def check(cls, v): ...
```
