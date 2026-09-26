# SEO Internal Linking Engine — project export

Verified against the latest decisions before packaging. See CHANGES.md.

## Contents

    wiki/                 13 wiki pages incl. ADRs.md — the source of truth
    github-import/         33 core-build issues, ready to import via gh CLI
    config/                pyproject.toml + .importlinter
    dev/                   local dev stack: compose, Makefile, corpus generator
    dev/scripts/corpus/    taxonomy.py, structure.py, content.py, embeddings.py,
                            writers.py — the corpus generation package
    diagrams/              jsx architecture diagrams + pitch/tech HTML

## Not included

Six superseded pre-wiki documents (SEO_Component_Reference.md through v5,
SEO_Functional_Process_Diagram.md, SEO_Internal_Linking_Engine_Architecture.md
and _v2, SEO_System_Architecture_Diagram.md) — everything in them is superseded
by wiki/. Ask if you want them included for history.

## Start here

wiki/Home.md -> wiki/Setup.md -> wiki/Core-Build-Plan.md

## Corpus package

    dev/scripts/corpus/taxonomy.py    topics, planted ground truth, CTR curve
    dev/scripts/corpus/structure.py   pages, body links, GSC metrics (deterministic)
    dev/scripts/corpus/content.py     template + LLM backends, constraint verification
    dev/scripts/corpus/embeddings.py  synthetic + Voyage backends, token batching
    dev/scripts/corpus/writers.py     Neo4j + Mongo persistence
    dev/scripts/generate_corpus.py    CLI entry point
    dev/scripts/verify_corpus.py      ground-truth assertion suite

    uv run python scripts/generate_corpus.py --pages 600         # instant
    uv run python scripts/verify_corpus.py                       # 9/9 checks
