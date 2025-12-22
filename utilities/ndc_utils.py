# File: src/rag_med/linkers/ndc_utils.py

import re
from typing import Optional

_ndc_digits = re.compile(r"\d+")


def normalize_ndc11(x: str) -> Optional[str]:
    """
    Normalize a variety of NDC formats to 11-digit numeric string (NDC11).
    Accepts: '1234-5678-90', '12345-6789-0', '01234567890', ' 1234-5678-90  '
    Returns None if cannot produce an 11-digit string.
    """
    if x is None:
        return None
    s = "".join(_ndc_digits.findall(str(x)))
    if len(s) == 10:
        s = s.zfill(11)
    if len(s) == 11 and s.isdigit():
        return s
    return None
