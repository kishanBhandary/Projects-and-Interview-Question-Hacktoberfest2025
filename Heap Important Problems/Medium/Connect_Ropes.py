# Connect Ropes to Minimize Cost

# Problem Statement:
# Given n ropes of different lengths, connect them into a single rope with minimum cost.
# The cost of connecting two ropes is equal to the sum of their lengths.

# Example:

# Input: ropes = [4,3,2,6]  
# Output: 29


# Python Code:

import heapq

def minCost(ropes):
    heapq.heapify(ropes)
    total_cost = 0
    while len(ropes) > 1:
        first = heapq.heappop(ropes)
        second = heapq.heappop(ropes)
        cost = first + second
        total_cost += cost
        heapq.heappush(ropes, cost)
    return total_cost