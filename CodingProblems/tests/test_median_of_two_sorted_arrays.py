import sys
import os
import unittest

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'algorithms')))

from median_of_two_sorted_arrays import findMedianSortedArrays

class TestFindMedianSortedArrays(unittest.TestCase):

    def test_example_1(self):
        self.assertEqual(findMedianSortedArrays([1, 3], [2]), 2.0)

    def test_example_2(self):
        self.assertEqual(findMedianSortedArrays([1, 2], [3, 4]), 2.5)

    def test_empty_first(self):
        self.assertEqual(findMedianSortedArrays([], [1]), 1.0)

    def test_empty_second(self):
        self.assertEqual(findMedianSortedArrays([1], []), 1.0)
        
    def test_different_lengths(self):
        self.assertEqual(findMedianSortedArrays([1, 2, 3], [4, 5, 6, 7, 8]), 4.5)

    def test_with_negatives(self):
        self.assertEqual(findMedianSortedArrays([-5, 3, 6, 12, 15], [-12, -10, -6, -3, 4, 10]), 3.0)

if __name__ == '__main__':
    unittest.main()
