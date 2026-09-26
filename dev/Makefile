SHELL := /bin/bash
CYPHER := docker exec -i neo4j cypher-shell -u neo4j -p localdevpassword

.PHONY: up down reset schema seed seed-real seed-prose seed-big report verify verify-real verify-db sanity bench eval eval-umap eval-stability projections drop-projections logs neo mongo

up:                ## Start neo4j + mongo, apply schema
	docker compose up -d
	@echo "waiting for neo4j..."
	@until docker exec neo4j cypher-shell -u neo4j -p localdevpassword "RETURN 1" >/dev/null 2>&1; do sleep 3; done
	@$(MAKE) schema
	@echo "ready  →  http://localhost:7474  (neo4j / localdevpassword)"

down:
	docker compose down

reset:             ## Wipe volumes and start clean
	docker compose down -v && $(MAKE) up

schema:
	@$(CYPHER) < scripts/01-schema.cypher

seed:              ## 600 pages, template content + synthetic vectors (instant)
	uv run python scripts/generate_corpus.py --pages 600 --seed 42

seed-real:         ## 600 pages, LLM prose + real Voyage embeddings (~15 min, costs money)
	uv run python scripts/generate_corpus.py --pages 600 --seed 42 \
		--content llm --embeddings voyage

seed-prose:        ## LLM prose, synthetic vectors — test HTML extraction without Voyage spend
	uv run python scripts/generate_corpus.py --pages 600 --seed 42 --content llm

verify:            ## Assert planted ground truth survived generation
	uv run python scripts/verify_corpus.py --pages 600

verify-real:       ## Same, against LLM-generated prose
	uv run python scripts/verify_corpus.py --pages 600 --content llm

seed-big:          ## 5000 pages, for performance work
	uv run python scripts/generate_corpus.py --pages 5000 --seed 42

report:            ## Print planted ground truth without touching the DBs
	uv run python scripts/generate_corpus.py --report

eval:              ## Leiden vs HDBSCAN, scored against ground truth
	uv run python scripts/eval_clustering.py

eval-umap:         ## Same, with UMAP dimensionality reduction first
	uv run python scripts/eval_clustering.py --umap

eval-stability:    ## How reproducible are HDBSCAN labels across runs?
	uv run python scripts/eval_clustering.py --stability 4

verify-db:         ## GDS version, available algorithms, vector function
	@$(CYPHER) < scripts/02-verify.cypher

sanity:            ## Confirm the seeded corpus looks right
	@$(CYPHER) < scripts/03-sanity.cypher

bench:             ## Leiden vs Louvain on the seeded graph
	@$(CYPHER) < scripts/04-clustering-bench.cypher


projections:       ## List held GDS projections (Community limit: 3)
	@$(CYPHER) "CALL gds.graph.list() YIELD graphName, nodeCount, memoryUsage RETURN graphName, nodeCount, memoryUsage;"

drop-projections:  ## Clear projections leaked by a failed run
	@$(CYPHER) "CALL gds.graph.list() YIELD graphName CALL gds.graph.drop(graphName) YIELD graphName AS d RETURN d;"

logs:
	docker compose logs -f

neo:
	docker exec -it neo4j cypher-shell -u neo4j -p localdevpassword

mongo:
	docker exec -it mongo mongosh linking_engine
