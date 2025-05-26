from pydantic import BaseModel
import logging

logger = logging.getLogger("defense_pipeline")

class AttackSchema(BaseModel):
    backdoor: bool
    evasion: bool
    poisoning: bool
    extraction: bool
    inference: bool
    other: bool