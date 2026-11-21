# Graph Traversal

Breadth-first search (BFS) explores a graph level by level using a queue. In an unweighted graph, BFS finds the shortest path from the source to every reachable node. Depth-first search (DFS) goes as deep as possible along each branch using a stack or recursion before backtracking.

Both BFS and DFS run in O(V + E) time with an adjacency list, where V is the number of vertices and E the number of edges. DFS is used for cycle detection, topological sorting and finding connected components.

Dijkstra's algorithm finds shortest paths in a graph with non-negative edge weights using a priority queue, in O((V + E) log V) time. Bellman-Ford handles negative edge weights and can detect negative cycles.
