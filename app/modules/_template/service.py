"""Business logic for the <module> module — kept independent of FastAPI."""

from .schemas import ExampleResponse


class ExampleService:
    async def run(self, payload: str) -> ExampleResponse:
        return ExampleResponse(result=payload.upper())
