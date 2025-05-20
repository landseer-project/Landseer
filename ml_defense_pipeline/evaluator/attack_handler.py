from pydantic import BaseModel

class AttackSchema(BaseModel):
    backdoor: bool
    evasion: bool
    poisoning: bool
    extraction: bool
    inference: bool
    other: bool

