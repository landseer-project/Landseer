from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class AttackTypes(BaseModel):
    backdoor: bool
    evasion: bool
    extraction: bool
    inference: bool
    other: bool

class AttackSchema(BaseModel):
    attacks: AttackTypes
