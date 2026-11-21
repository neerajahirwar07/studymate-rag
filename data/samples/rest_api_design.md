# REST API Design

REST is an architectural style where resources are identified by URLs and manipulated using standard HTTP methods. GET reads a resource, POST creates one, PUT replaces one, PATCH updates part of one, and DELETE removes one.

A REST API is stateless: every request carries all the information the server needs, so servers can scale horizontally. Idempotent methods such as GET, PUT and DELETE give the same result when repeated. POST is not idempotent.

Status codes describe the outcome. 200 means success, 201 means a resource was created, 400 means a bad request, 401 means unauthenticated, 403 means forbidden, 404 means not found, and 500 means a server error. Good APIs also validate input, paginate large lists and version their endpoints.
