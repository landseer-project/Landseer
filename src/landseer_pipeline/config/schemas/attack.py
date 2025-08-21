from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class AttackTypes(BaseModel):
    backdoor: bool
    adversarial: bool
    outlier: bool
    inference: bool
    other: bool

class AttackSchema(BaseModel):
    attacks: AttackTypes
