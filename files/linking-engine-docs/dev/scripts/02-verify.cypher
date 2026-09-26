RETURN gds.version() AS gds_version;

CALL gds.list() YIELD name
WHERE name CONTAINS 'leiden' OR name CONTAINS 'louvain'
   OR name CONTAINS 'pageRank' OR name CONTAINS 'betweenness'
RETURN name ORDER BY name;

RETURN vector.similarity.cosine([1.0,0.0],[1.0,0.0]) AS should_be_1;

CALL gds.graph.list() YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;
