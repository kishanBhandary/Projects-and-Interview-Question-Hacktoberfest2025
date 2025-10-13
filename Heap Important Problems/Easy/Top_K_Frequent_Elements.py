# Top K Frequent Elements

# Problem Statement:
# Given an integer array nums and an integer k, return the k most frequent elements.
# Use a heap for efficient frequency tracking.

# Example:

# Input: nums = [1,1,1,2,2,3], k = 2  
# Output: [1,2]


# Python Code:

import heapq
from collections import Counter

def topKFrequent(nums, k):
    count = Counter(nums)
    return [item for item, freq in heapq.nlargest(k, count.items(), key=lambda x: x[1])]
