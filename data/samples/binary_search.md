# Binary Search

Binary search finds a target value in a sorted array by repeatedly halving the search range. It compares the target with the middle element. If they are equal, the search ends. If the target is smaller, the search continues in the left half. If the target is larger, it continues in the right half.

The time complexity of binary search is O(log n) because the range shrinks by half at every step. The space complexity is O(1) for the iterative version and O(log n) for the recursive version because of the call stack.

Binary search only works when the array is sorted. A common bug is computing the middle index as (low + high) / 2, which can overflow for very large indices. The safe form is low + (high - low) / 2.
