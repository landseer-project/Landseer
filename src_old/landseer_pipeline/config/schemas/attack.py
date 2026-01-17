from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class AttackTypes(BaseModel):
    backdoor: bool          # Backdoor/trojan attacks
    adversarial: bool       # PGD, FGSM attacks  
    outlier: bool           # OOD detection
    carlini: bool           # Carlini & Wagner L2 attacks
    watermarking: bool = False      # Watermark detection attacks
    fingerprinting: bool = False    # Model fingerprinting attacks
    inference: bool         # Privacy/membership inference attacks
    other: bool             # Other miscellaneous attacks

class AttackSchema(BaseModel):
    attacks: AttackTypes
