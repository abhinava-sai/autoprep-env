from pydantic import BaseModel
from typing import Optional


class Observation(BaseModel):
    missing_values: int
    duplicate_rows: int
    outliers: int
    step_count: int


class Action(BaseModel):
    action_type: str


class Reward(BaseModel):
    value: float
    reason: Optional[str] = None