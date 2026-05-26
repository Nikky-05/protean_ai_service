"""HTTP routes for the <module> module."""

from fastapi import APIRouter, Depends

from .schemas import ExampleRequest, ExampleResponse
from .service import ExampleService

router = APIRouter()


def _service() -> ExampleService:
    return ExampleService()


@router.post("/example", response_model=ExampleResponse)
async def example(
    body: ExampleRequest, svc: ExampleService = Depends(_service)
) -> ExampleResponse:
    return await svc.run(body.payload)
