def _multiplier(s: str) -> int:
    if s == "gb":
        return 1024 * 1024 * 1024
    elif s == "mb":
        return 1024 * 1024
    elif s == "kb":
        return 1024
    else:
        raise Exception(f"Invalid size string {s}")


def human2bytes(s: str) -> int:
    multiplier = _multiplier(s.strip().lower()[-2:])
    n = float(s.strip().lower()[:-2].strip())
    return round(n * multiplier)
