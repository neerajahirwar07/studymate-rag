# Hash Tables

A hash table stores key-value pairs and gives average O(1) time for insert, lookup and delete. A hash function converts a key into an index in an array of buckets.

When two different keys map to the same bucket, a collision occurs. Chaining resolves collisions by storing a linked list of entries in each bucket. Open addressing resolves them by probing for the next free slot using linear probing, quadratic probing or double hashing.

The load factor is the number of stored entries divided by the number of buckets. When the load factor grows too high, the table is resized and all keys are rehashed. With many collisions, the worst case lookup time degrades to O(n).
