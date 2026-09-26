// Replace the legacy global uniqueness with per-tenant uniqueness.
DROP CONSTRAINT page_url IF EXISTS;
DROP CONSTRAINT keyword_text_lang IF EXISTS;
CREATE CONSTRAINT page_tenant_url IF NOT EXISTS
  FOR (p:Page) REQUIRE (p.tenantId, p.url) IS UNIQUE;
CREATE CONSTRAINT keyword_tenant_text_lang IF NOT EXISTS
  FOR (k:Keyword) REQUIRE (k.tenantId, k.text, k.language) IS UNIQUE;

CREATE INDEX page_tenant IF NOT EXISTS FOR (p:Page) ON (p.tenantId);
CREATE INDEX keyword_tenant IF NOT EXISTS FOR (k:Keyword) ON (k.tenantId);
CREATE INDEX page_topic IF NOT EXISTS FOR (p:Page) ON (p.topic);
CREATE INDEX page_indexable IF NOT EXISTS FOR (p:Page) ON (p.isIndexable);
CREATE INDEX page_lifecycle IF NOT EXISTS FOR (p:Page) ON (p.lifecycleStage);
CREATE INDEX page_link_community IF NOT EXISTS FOR (p:Page) ON (p.linkCommunityId);
CREATE INDEX page_kw_community IF NOT EXISTS FOR (p:Page) ON (p.keywordCommunityId);
CREATE INDEX link_audited IF NOT EXISTS FOR ()-[r:LINKS_TO]-() ON (r.auditedAt);
CREATE INDEX link_anchor_type IF NOT EXISTS FOR ()-[r:LINKS_TO]-() ON (r.anchorType);
