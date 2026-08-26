---
name: py-validation
description: Use this skill when adding validation to Python API payloads and response models. Covers Pydantic v2 model definitions, field_validator usage, and response_model wiring for FastAPI routes.
---

# Python payload validation

Validate API payloads with Pydantic v2 models rather than hand-rolled checks.

## Rules

1. One `BaseModel` subclass per payload shape, every field annotated.
2. Use `@field_validator` with `@classmethod`; the legacy `@validator` is gone in v2.
3. FastAPI routes declare `response_model=` so the schema is enforced.

## Example

```python
from pydantic import BaseModel, field_validator


class Payload(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("name must not be empty")
        return value
```
