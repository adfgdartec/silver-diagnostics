from typing import Any


def is_finite(value: Any) -> bool:
    try:
        return value == value and abs(value) != float("inf")
    except (TypeError, ValueError):
        return False
