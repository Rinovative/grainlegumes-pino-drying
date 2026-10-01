# Project workflow

`.agents/project/` holds portable project knowledge. Canonical, versioned JSON records are `project.json` (identity and reconciled task outcomes), `tasks/` (task contracts), and `decisions/` (durable decisions). This README and `.gitignore` are also versioned workflow documentation/configuration.

[OVERVIEW.md](OVERVIEW.md) is a compact, generated dashboard. [NEXT_ROUND.md](../../../runtime/codex/exports/NEXT_ROUND.md) is the richer planning handoff for ordinary ChatGPT, covering unresolved decisions, candidate work, and launch context. Both are derived from canonical state, not independent sources of truth.

`OVERVIEW.md` and the host-local binding `local.json` are ignored. Detailed attempts, runtime evidence, source copies, patches, dated handoffs, and `NEXT_ROUND.md` remain outside Git in the runtime selected by `local.json`. The packet link above uses the standard outer-workspace layout.

## Normal cycle

1. Inspect current state if useful. Ask Codex in natural language to show a task, decision, overview, or next-round packet; it can use the helpers internally.
2. Take the current `NEXT_ROUND.md` into ordinary ChatGPT for discussion.
3. Choose the next work and obtain complete Codex launch prompts.
4. Launch one root with bounded delegation, or multiple independent roots with separate assignments and isolated writing copies.
5. Workers return scoped handoffs with results, validation, and unresolved findings.
6. The final reconciler reconciles selected results, integrates authorized changes, validates the combined result, updates canonical state, and regenerates `OVERVIEW.md` and `NEXT_ROUND.md`. A single root may reconcile its own deliverable; workers do not publish competing project packets.

Readiness is not execution authorization: the user selects the work and scientific priorities.

Manual inspection, recovery, and exceptional workflow commands are documented in the shared project-workflow guide: `$HOME/.agents/skills/project-workflow/references/guide.md`.
