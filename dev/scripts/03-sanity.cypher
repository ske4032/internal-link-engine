// Does the seeded corpus look right?

MATCH (p:Page) RETURN 'pages' AS what, count(*) AS n
UNION ALL MATCH ()-[l:LINKS_TO]->() RETURN 'links', count(l)
UNION ALL MATCH (k:Keyword) RETURN 'keywords', count(k)
UNION ALL MATCH ()-[t:TARGETS_KEYWORD]->() RETURN 'targets_keyword', count(t);

// Planted issues the audit must find
MATCH ()-[l:LINKS_TO]->()
RETURN l.anchorType AS type, l.linkPosition AS pos,
       l.isFollow AS follow, l.targetHttpStatus AS status, count(*) AS n
ORDER BY n DESC;

// Orphans — should be exactly the planted NEW pages
MATCH (p:Page) WHERE NOT ()-[:LINKS_TO]->(p)
RETURN p.url, p.lifecycleStage, p.topic;

// Do synthetic embeddings cluster by topic?
MATCH (a:Page {pageType:'PILLAR'}), (b:Page {pageType:'PILLAR'})
WHERE a.url < b.url
RETURN a.topic, b.topic,
       round(vector.similarity.cosine(a.content_embedding, b.content_embedding), 3) AS sim
ORDER BY sim DESC;
