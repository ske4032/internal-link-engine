// Runs once on first container start.
const db = db.getSiblingDB('linking_engine');

const collections = [
  'pages', 'gsc_metrics', 'gsc_queries', 'strategic_keywords',
  'link_audit', 'recommendations', 'anchor_feedback',
  'tenant_config', 'ctr_curves'
];
collections.forEach(c => db.createCollection(c));

// Every collection is tenant-scoped. Compound indexes lead with tenantId.
db.pages.createIndex({ tenantId: 1, url: 1 }, { unique: true });
db.pages.createIndex({ tenantId: 1, crawledAt: -1 });
db.pages.createIndex({ tenantId: 1, contentHash: 1 });

db.gsc_metrics.createIndex({ tenantId: 1, url: 1 }, { unique: true });
db.gsc_queries.createIndex({ tenantId: 1, url: 1, impressions: -1 });
db.gsc_queries.createIndex({ tenantId: 1, query: 1 });

db.strategic_keywords.createIndex({ tenantId: 1, url: 1 });
db.strategic_keywords.createIndex({ tenantId: 1, keyword: 1 });
db.strategic_keywords.createIndex(
  { tenantId: 1, url: 1, isPrimary: 1 },
  { partialFilterExpression: { isPrimary: true } }
);

db.link_audit.createIndex({ tenantId: 1, sourceUrl: 1, targetUrl: 1 });
db.link_audit.createIndex({ tenantId: 1, auditedAt: -1 });

db.recommendations.createIndex({ tenantId: 1, fromUrl: 1, score: -1 });
db.recommendations.createIndex({ tenantId: 1, actionType: 1, score: -1 });
db.recommendations.createIndex({ tenantId: 1, createdAt: 1 },
  { expireAfterSeconds: 2592000 });   // 30d TTL, matches SUGGESTED_ACTION cleanup

db.anchor_feedback.createIndex({ tenantId: 1, createdAt: -1 });
db.anchor_feedback.createIndex({ tenantId: 1, actionType: 1, accepted: 1 });

db.tenant_config.createIndex({ tenantId: 1 }, { unique: true });
db.ctr_curves.createIndex({ tenantId: 1 }, { unique: true });

// Seed the demo tenant
db.tenant_config.insertOne({
  tenantId: 'demo',
  neo4jUri: 'bolt://neo4j-demo:7687',
  embeddingProvider: 'VOYAGE_API',
  embeddingModel: 'voyage-4-large',
  embeddingDimensions: 1024,
  embeddingLocalFallbackModel: 'voyage-4-nano',
  contentGapModel: 'qwen3.5:4b',
  contentGapMinPriority: 4,
  contentGapMinOpportunity: 500,
  anchorTypeProfile: { exact: 0.15, partial: 0.20, natural: 0.50, branded: 0.15 },
  auditEnabled: true,
  discoveryEnabled: false,
  siteLanguages: ['en'],
  primaryLanguage: 'en',
  clientTier: 'STARTER',
  lifecycleBoostEnabled: true,
  reservedNewPageSlotPct: 0.15,
  diversityCapPct: 0.15,
  maxRecommendationsPerSource: 10,
  createdAt: new Date()
});

print('linking_engine initialised: ' + collections.length + ' collections');
