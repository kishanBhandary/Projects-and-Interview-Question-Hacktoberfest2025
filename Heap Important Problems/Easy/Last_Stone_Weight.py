# Last Stone Weight

# Problem Statement:
# You are given an array stones where each stone has a positive integer weight.
# Each turn, smash the two heaviest stones together. Return the weight of the last remaining stone (or 0 if none remain).

# Example:

# Input: stones = [2,7,4,1,8,1]  
# Output: 1


# Python Code:

import heapq

def lastStoneWeight(stones):
    stones = [-s for s in stones]  # max heap simulation
    heapq.heapify(stones)
    while len(stones) > 1:
        first = -heapq.heappop(stones)
        second = -heapq.heappop(stones)
        if first != second:
            heapq.heappush(stones, -(first - second))
    return -stones[0] if stones else 0