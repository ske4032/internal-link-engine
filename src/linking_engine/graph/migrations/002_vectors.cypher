// Dimension is fixed at creation; GraphRepo.migrate verifies 2048 afterwards.
CREATE VECTOR INDEX page_content IF NOT EXISTS
  FOR (p:Page) ON p.content_embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 2048,
    `vector.similarity_function`: 'cosine' }};
CREATE VECTOR INDEX page_gnn IF NOT EXISTS
  FOR (p:Page) ON p.gnn_embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 2048,
    `vector.similarity_function`: 'cosine' }};
