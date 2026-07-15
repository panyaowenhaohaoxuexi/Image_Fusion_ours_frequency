"""Normalization helpers shared by the enhanced fusion components."""


def get_valid_group_count(channels: int) -> int:
    """Return a GroupNorm group count that divides ``channels`` exactly."""
    if channels < 1:
        raise ValueError(f"channels must be positive, got {channels}.")
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
