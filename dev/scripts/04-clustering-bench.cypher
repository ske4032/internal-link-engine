// Compare Leiden and Louvain on the same projection.
// Community Edition: 3 projections max, 4-core concurrency cap.

CALL gds.graph.drop('link-graph', false) YIELD graphName;
CALL gds.graph.project('link-graph', 'Page', {LINKS_TO: {orientation: 'UNDIRECTED'}});

CALL gds.leiden.stats('link-graph', {concurrency: 4, randomSeed: 42})
YIELD communityCount, modularity, ranLevels
RETURN 'leiden' AS algo, communityCount, modularity, ranLevels;

CALL gds.louvain.stats('link-graph', {concurrency: 4})
YIELD communityCount, modularity, ranLevels
RETURN 'louvain' AS algo, communityCount, modularity, ranLevels;

// Write Leiden, then check whether clusters recover the planted topics.
CALL gds.leiden.write('link-graph',
  {writeProperty:'linkCommunityId', concurrency:4, randomSeed:42})
YIELD communityCount;

MATCH (p:Page)
RETURN p.linkCommunityId AS community, p.topic AS planted_topic, count(*) AS n
ORDER BY community, n DESC;

CALL gds.graph.drop('link-graph') YIELD graphName;
