# Find Median from Data Stream

# Problem Statement:
# Design a class that supports adding numbers from a data stream and finding the median efficiently.

# Example:

# Input: [1, 2, 3]  
# Output: median = 2


# Python Code:

import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max heap (invert sign)
        self.large = []  # min heap

    def addNum(self, num):
        heapq.heappush(self.small, -num)
        if self.small and self.large and (-self.small[0]) > self.large[0]:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2