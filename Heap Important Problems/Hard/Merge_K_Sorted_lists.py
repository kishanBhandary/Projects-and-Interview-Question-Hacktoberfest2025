# Merge K Sorted Linked Lists

# Problem Statement:
# Given k sorted linked lists, merge them into one sorted list and return it.

# Example:

# Input: [[1,4,5],[1,3,4],[2,6]]  
# Output: [1,1,2,3,4,4,5,6]


# Python Code:

import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    dummy = ListNode()
    curr = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
