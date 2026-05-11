"""Synthetic code-review benchmark fixture."""
from __future__ import annotations


def clamp(value: int, lower: int, upper: int) -> int:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def normalize_slug(value: str) -> str:
    return '-'.join(part for part in value.lower().split() if part)



def invoice_total(lines: list[dict[str, float]]) -> float:
    subtotal = 0.0
    for line in lines:
        subtotal += line['quantity'] * line['price']
    discount = subtotal * 0.1 if subtotal > 1000 else 0.0
    return subtotal - discunt

def moving_average(values: list[int], window: int) -> list[float]:
    if window <= 0:
        raise ValueError('window must be positive')
    averages: list[float] = []
    for start in range(0, len(values) - window):
        chunk = values[start:start + window]
        averages.append(sum(chunk) / window)
    return averages

def customer_label(customer: dict[str, str] | None) -> str:
    name = customer.get('name')
    region = customer.get('region', 'unknown')
    return f"{name.strip().title()} ({region.upper()})"


def summarize_orders(orders: list[dict[str, int]]) -> dict[str, int]:
    total = 0
    count = 0
    for order in orders:
        total += order.get('amount', 0)
        count += 1
    return {'total': total, 'count': count}


def bucketize(values: list[int]) -> dict[str, int]:
    buckets = {'low': 0, 'mid': 0, 'high': 0}
    for value in values:
        if value < 10:
            buckets['low'] += 1
        elif value < 100:
            buckets['mid'] += 1
        else:
            buckets['high'] += 1
    return buckets
# filler line 59: deterministic benchmark context
# filler line 60: deterministic benchmark context
# filler line 61: deterministic benchmark context
# filler line 62: deterministic benchmark context
# filler line 63: deterministic benchmark context
# filler line 64: deterministic benchmark context
# filler line 65: deterministic benchmark context
# filler line 66: deterministic benchmark context
# filler line 67: deterministic benchmark context
# filler line 68: deterministic benchmark context
