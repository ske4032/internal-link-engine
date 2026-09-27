# Agent instructions: Internal Linking Intelligence Engine

## Code search and context (mandatory)

1. **cxpak first.** Before reading or changing code, get context from cxpak: the MCP tools
   (`cxpak_context` with `op` `context_for_task`, `search` or `overview`; `cxpak_graph` for
   references, paths and blast radius) or the CLI (`cxpak search --op references --symbol X .`,
   `cxpak trace <symbol> .`). Open whole files only after cxpak has narrowed the set.
2. **Exact text: `rg`, never `grep`.** The built-in Grep tool already runs ripgrep; in the shell use `rg`.
3. **Search by meaning: `zg`.** `zg query "<intent>"` or the MCP tool `zvec_grep_search`. Confirm a hit
   with `rg` or cxpak before editing. After large changes, refresh the index with `zg index`
   (stored in `.zvec-grep/`, gitignored).

## Tenant isolation (mandatory)

Tenant data is never shared. Every read, cache lookup, deduplication, reuse and write is
filtered by `tenantId`: no cross-tenant reuse of vectors, anchors, sentences or
configuration, even for identical text.
