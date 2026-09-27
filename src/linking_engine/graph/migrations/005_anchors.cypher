// One Anchor node per normalised anchor text per tenant; LINKS_TO.anchorKey joins to it.
CREATE CONSTRAINT anchor_tenant_text IF NOT EXISTS
  FOR (a:Anchor) REQUIRE (a.tenantId, a.text) IS UNIQUE;
// Looks up an existing sentence vector by content before paying for a new one.
CREATE INDEX link_surrounding_hash IF NOT EXISTS
  FOR ()-[r:LINKS_TO]-() ON (r.surroundingEmbeddedHash);
