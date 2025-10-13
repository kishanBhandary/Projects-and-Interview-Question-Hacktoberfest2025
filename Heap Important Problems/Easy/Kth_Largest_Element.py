# Problem Statement:
# Given an integer array nums and an integer k, return the kth largest element in the array.
# You must use a heap-based approach.

# Example:

# Input: nums = [3,2,1,5,6,4], k = 2  
# Output: 5


# Python Code:

import heapq

def findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]