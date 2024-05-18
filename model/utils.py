from pydantic import BaseModel

class GenerationRequest(BaseModel):
    seed: int
    text: str