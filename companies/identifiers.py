"""Telling what a search query is: an LEI, an ISIN, a register number or a name.

LEIs and ISINs carry check digits, so a query that passes the check is
almost certainly one; everything else is tried as a ticker, a number or a
name by each source.
"""
from __future__ import annotations

import re

LEI = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")
ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
UK_COMPANY_NUMBER = re.compile(r"^(?:[A-Z]{2}\d{6}|\d{8})$")


def _alnum_to_digits(text: str) -> str:
    return "".join(str(int(ch, 36)) for ch in text)


def is_lei(value: str) -> bool:
    """ISO 17442: 20 characters whose number (letters as 10-35) is 1 mod 97."""
    v = value.strip().upper()
    return bool(LEI.match(v)) and int(_alnum_to_digits(v)) % 97 == 1


def is_isin(value: str) -> bool:
    """ISO 6166: 12 characters, the last a Luhn check over the others' digits."""
    v = value.strip().upper()
    if not ISIN.match(v):
        return False
    digits = _alnum_to_digits(v)
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return total % 10 == 0


def is_uk_company_number(value: str) -> bool:
    v = value.strip().upper().replace(" ", "")
    return bool(UK_COMPANY_NUMBER.match(v.zfill(8) if v.isdigit() and len(v) >= 6 else v))
