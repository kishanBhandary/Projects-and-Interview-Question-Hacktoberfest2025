# Sliding Window Maximum

# Problem Statement:
# Given an integer array nums and a window size k, return the maximum value in each sliding window.

# Example:

# Input: nums = [1,3,-1,-3,5,3,6,7], k = 3  
# Output: [3,3,5,5,6,7]


# Python Code:

import heapq

def maxSlidingWindow(nums, k):
    res, heap = [], []
    for i, n in enumerate(nums):
        heapq.heappush(heap, (-n, i))
        while heap[0][1] <= i - k:
            heapq.heappop(heap)
        if i >= k - 1:
            res.append(-heap[0][0])
    return res
