```markdown
# 🧱 Heap Problems (Min Heap / Max Heap)

This folder contains **9 curated Heap problems** categorized by difficulty — _Easy_, _Medium_, and _Hard_.  
Each problem demonstrates how to use **priority queues (min-heap / max-heap)** efficiently in Python using the built-in `heapq` module.

---

## 📁 Folder Structure
```

Heap_Problems/
├── README.md
├── Easy/
│ ├── Kth_Largest_Element.py
│ ├── Top_K_Frequent_Elements.py
│ └── Last_Stone_Weight.py
├── Medium/
│ ├── K_Sorted_Array.py
│ ├── Connect_Ropes.py
│ └── Find_Median_from_Data_Stream.py
├── Hard/
│ ├── Merge_K_Sorted_Lists.py
│ ├── Sliding_Window_Maximum.py
│ └── Reorganize_String.py

```

---

## 🟢 Easy Level Problems

| No. | Problem | Concept |
|-----|----------|----------|
| 1 | **Kth Largest Element in an Array** | Use `heapq.nlargest()` or maintain a min-heap of size `k`. |
| 2 | **Top K Frequent Elements** | Count frequency and use a max heap to get top K. |
| 3 | **Last Stone Weight** | Simulate stone smashing using a max heap. |

---

## 🟡 Medium Level Problems

| No. | Problem | Concept |
|-----|----------|----------|
| 4 | **Sort a K-Sorted Array** | Use a min heap to place elements in correct order efficiently. |
| 5 | **Connect Ropes to Minimize Cost** | Always connect two smallest ropes first using a min heap. |
| 6 | **Find Median from Data Stream** | Maintain two heaps (max + min) for balanced median tracking. |

---

## 🔴 Hard Level Problems

| No. | Problem | Concept |
|-----|----------|----------|
| 7 | **Merge K Sorted Linked Lists** | Use a min heap to efficiently merge k sorted streams. |
| 8 | **Sliding Window Maximum** | Maintain a max heap with window indices for each position. |
| 9 | **Reorganize String** | Use a max heap to arrange characters so no two are adjacent. |

---

## 🧠 Concepts Covered
- Min Heap & Max Heap
- Priority Queue Operations
- Frequency Counting
- Sliding Window with Heap
- Stream Processing
- Greedy Algorithms using Heaps

---

## 🐍 Tech Stack
- **Language:** Python
- **Module:** `heapq` (for heap operations), `collections` (for counters and frequency maps)

---


## 💡 Why This Section Matters
- Builds a strong foundation in **priority queue-based problem solving**.
- Enhances understanding of how heaps optimize time complexity in sorting and searching tasks.
- Aids contributors in learning clean, efficient implementations for **coding interviews** and **competitive programming**.

---
```
