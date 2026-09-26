// Rename legacy httpStatus / targetHttpStatus to statusCode / targetStatusCode.
MATCH (p:Page) WHERE p.httpStatus IS NOT NULL
CALL (p) {
  SET p.statusCode = p.httpStatus
  REMOVE p.httpStatus
} IN TRANSACTIONS OF 1000 ROWS;
MATCH ()-[r:LINKS_TO]->() WHERE r.targetHttpStatus IS NOT NULL
CALL (r) {
  SET r.targetStatusCode = r.targetHttpStatus
  REMOVE r.targetHttpStatus
} IN TRANSACTIONS OF 1000 ROWS;
