// One Hub per HDBSCAN cluster per tenant; Page.hubId joins to it. Retired hubs stay so ids are never reused.
CREATE CONSTRAINT hub_tenant_id IF NOT EXISTS
  FOR (h:Hub) REQUIRE (h.tenantId, h.hubId) IS UNIQUE;
