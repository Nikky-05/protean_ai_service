"""Request / response models for the <module> module."""

from pydantic import BaseModel


class ExampleRequest(BaseModel):
    payload: str


class ExampleResponse(BaseModel):
    result: str
