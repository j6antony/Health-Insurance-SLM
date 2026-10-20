"""Whitespace based reading order, retaining every original line and its metadata."""


def order_lines(lines):
    if len(lines) < 2 or any(not line.get('bbox') for line in lines):
        return list(lines)

    def gaps(items, axis):
        intervals = sorted((l['bbox'][axis], l['bbox'][axis + 2]) for l in items)
        end = intervals[0][1]
        found = []
        for start, stop in intervals[1:]:
            if start > end:
                found.append((start - end, (start + end) / 2))
            end = max(end, stop)
        return found

    # A vertical gutter defines independent columns. Full-width lines are
    # separated into horizontal bands before column detection is retried.
    vertical = [g for g in gaps(lines, 0) if g[0] >= 18]
    horizontal = [g for g in gaps(lines, 1) if g[0] >= 3]
    if vertical:
        axis, (_, cut) = 0, max(vertical)
    elif horizontal:
        axis, (_, cut) = 1, max(horizontal)
    else:
        return sorted(lines, key=lambda l: (round(l['bbox'][1] / 3), l['bbox'][0]))
    first = [l for l in lines if l['bbox'][axis + 2] < cut]
    second = [l for l in lines if l['bbox'][axis] > cut]
    return order_lines(first) + order_lines(second)
