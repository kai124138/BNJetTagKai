# AGENTS.md

If you are an AI coding agent (Claude, Codex, Cursor, Copilot, etc.), read
**[`RULES.md`](RULES.md)** before doing anything. It is the binding contract for
this repo and applies on every session, even if the user's prompt doesn't
mention it.

The short version (full details and the mandatory checklist are in `RULES.md`):

1. Orient first: read `RULES.md`, `README.md`, `docs/NEXT_STEPS.md`, and recent
   `docs/CHANGELOG.md` entries; run `git log --oneline -15` and `git status`.
2. Make a focused, style-matching change. Ask before guessing on ambiguity.
3. Verify what you can; clearly state what you could NOT verify.
4. **Update the docs your change affects — no stale numbers.**
5. **Append a dated entry to `docs/CHANGELOG.md`** (what / why / verified /
   still-broken). Mandatory.
6. **Update `docs/NEXT_STEPS.md`** to reflect reality after your change.
   Mandatory.
7. Use Conventional Commit messages with a real body. **Do not push without
   permission.**

You are not done until the `RULES.md` self-check boxes are all true.
