CREATE CONSTRAINT page_url IF NOT EXISTS
  FOR (p:Page) REQUIRE p.url IS UNIQUE;
CREATE CONSTRAINT keyword_text_lang IF NOT EXISTS
  FOR (k:Keyword) REQUIRE (k.text, k.language) IS UNIQUE;

CREATE INDEX page_topic IF NOT EXISTS FOR (p:Page) ON (p.topic);
CREATE INDEX page_indexable IF NOT EXISTS FOR (p:Page) ON (p.isIndexable);
CREATE INDEX page_lifecycle IF NOT EXISTS FOR (p:Page) ON (p.lifecycleStage);
CREATE INDEX page_link_community IF NOT EXISTS FOR (p:Page) ON (p.linkCommunityId);
CREATE INDEX page_kw_community IF NOT EXISTS FOR (p:Page) ON (p.keywordCommunityId);
CREATE INDEX link_audited IF NOT EXISTS FOR ()-[l:LINKS_TO]-() ON (l.auditedAt);
CREATE INDEX action_type IF NOT EXISTS FOR ()-[s:SUGGESTED_ACTION]-() ON (s.actionType);

// Dimension is fixed at creation. Changing 1024 later means dropping the
// index, re-embedding every page, and retraining the GNN.
CREATE VECTOR INDEX page_content IF NOT EXISTS
  FOR (p:Page) ON p.content_embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 2048, `vector.similarity_function`: 'cosine' }};
CREATE VECTOR INDEX page_gnn IF NOT EXISTS
  FOR (p:Page) ON p.gnn_embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 2048, `vector.similarity_function`: 'cosine' }};
