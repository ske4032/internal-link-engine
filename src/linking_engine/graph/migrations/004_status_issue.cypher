// Mark non-2xx pages and flag links pointing at them as FIX.
CREATE INDEX page_status_issue IF NOT EXISTS FOR (p:Page) ON (p.statusIssue);
CREATE INDEX link_verdict IF NOT EXISTS FOR ()-[r:LINKS_TO]-() ON (r.verdict);
MATCH (p:Page) WHERE p.statusCode >= 300
CALL (p) {
  SET p.statusIssue = CASE WHEN p.statusCode < 400 THEN 'REDIRECTED' ELSE 'BROKEN' END
} IN TRANSACTIONS OF 1000 ROWS;
MATCH ()-[r:LINKS_TO]->(t:Page) WHERE t.statusIssue IS NOT NULL
CALL (r, t) {
  SET r.targetStatusCode = t.statusCode,
      r.issueFlags = [f IN coalesce(r.issueFlags, []) WHERE NOT f IN ['BROKEN', 'REDIRECTED']] + [t.statusIssue],
      r.verdict = 'FIX'
} IN TRANSACTIONS OF 1000 ROWS;
