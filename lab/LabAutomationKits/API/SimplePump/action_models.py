from pydantic import BaseModel
from typing import Optional

class PumpCommand(BaseModel):
    state: bool = False
    speed: int = 2000
    dir: bool = True

class ActionRequest(BaseModel):
    id: str
    time: int = 10000  # ms
    pumpA: Optional[PumpCommand] = None 