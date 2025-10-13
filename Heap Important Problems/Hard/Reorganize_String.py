# Reorganize String

# Problem Statement:
# Rearrange the characters of a string so that no two adjacent characters are the same.
# If not possible, return an empty string.

# Example:

# Input: s = "aab"  
# Output: "aba"


# Python Code:

import heapq
from collections import Counter

def reorganizeString(s):
    count = Counter(s)
    max_heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(max_heap)

    prev = None
    result = []

    while max_heap or prev:
        if not max_heap and prev:
            return ""  # not possible
        freq, char = heapq.heappop(max_heap)
        result.append(char)
        if prev:
            heapq.heappush(max_heap, prev)
            prev = None
        if freq + 1 < 0:
            prev = (freq + 1, char)

    return "".join(result)