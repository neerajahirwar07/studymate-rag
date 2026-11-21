# Database Normalization

Normalization organizes a relational schema to reduce redundancy and avoid update, insert and delete anomalies. It works by decomposing tables using functional dependencies.

First normal form (1NF) requires that every column holds atomic values and there are no repeating groups. Second normal form (2NF) requires 1NF and that every non-key attribute depends on the whole primary key, which removes partial dependencies. Third normal form (3NF) requires 2NF and that no non-key attribute depends on another non-key attribute, which removes transitive dependencies.

Boyce-Codd normal form (BCNF) is stricter than 3NF: for every functional dependency X to Y, X must be a superkey. Denormalization deliberately adds redundancy back to speed up read-heavy queries.
