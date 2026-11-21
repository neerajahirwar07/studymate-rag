# ACID Transactions

A transaction is a group of database operations that must succeed or fail as one unit. ACID describes four guarantees of a reliable transaction system.

Atomicity means all operations in a transaction are applied, or none are. Consistency means a transaction moves the database from one valid state to another and never violates constraints. Isolation means concurrent transactions do not see each other's partial changes. Durability means that once a transaction is committed, its changes survive crashes, usually through a write-ahead log.

Isolation levels trade correctness for speed. Read uncommitted allows dirty reads. Read committed prevents dirty reads. Repeatable read prevents non-repeatable reads. Serializable prevents phantom reads and behaves as if transactions ran one at a time.
