# Sort a K-Sorted Array

# Problem Statement:
# Given an array where each element is at most k positions away from its sorted position,
# sort the array efficiently using a heap.

# Example:

# Input: arr = [6,5,3,2,8,10,9], k = 3  
# Output: [2,3,5,6,8,9,10]


# Python Code:

import heapq

def sortKSortedArray(arr, k):
    heap = arr[:k + 1]
    heapq.heapify(heap)
    idx = 0
    for i in range(k + 1, len(arr)):
        arr[idx] = heapq.heappop(heap)
        heapq.heappush(heap, arr[i])
        idx += 1
    while heap:
        arr[idx] = heapq.heappop(heap)
        idx += 1
    return arr
