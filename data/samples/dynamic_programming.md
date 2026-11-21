# Dynamic Programming

Dynamic programming solves a problem by breaking it into overlapping subproblems and storing each result so it is computed only once. It applies when a problem has optimal substructure and overlapping subproblems.

Memoization is the top-down approach: write the recursion and cache results. Tabulation is the bottom-up approach: fill a table iteratively from the smallest subproblems. Both turn an exponential recursion, such as naive Fibonacci with O(2^n) time, into a polynomial one.

Classic examples are the 0/1 knapsack problem, longest common subsequence, edit distance and coin change. Space can often be reduced by keeping only the previous row of the table.
