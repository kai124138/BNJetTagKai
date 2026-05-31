# RULES.md — Read This First

**Everyone who touches this repo reads this file before doing anything else.**
That means humans, and it especially means LLMs/agents (Claude, Codex, Cursor,
Copilot, etc.). If you are an AI assistant, treat this as a binding contract:
follow it on every session, even if the user's prompt doesn't mention it.

The point of this file is **continuity**. This is a research repo worked on in
short bursts, often by a different person or a different LLM each time. The only
way the next person isn't lost is if the current person leaves a clean trail.
Code without that trail is half-finished work.

---

## The Prime Directive

> **Never end a work session without updating the written record.**

If you changed code, you change docs. If you discovered something, you write it
down. If you left something unfinished, you say so explicitly and say what's
next. An LLM that fixes a bug but doesn't record *what* it fixed, *why*, and
*what's still broken* has done half the job and created a trap for the next one.

---

## Every session: the required workflow

Do these in order. Steps 1, 6, and 7 are **mandatory** — never skip them.

### 1. Orient before you touch anything
- Read `RULES.md` (this file), `README.md`, and `docs/NEXT_STEPS.md`.
- Skim the most recent `docs/session_summary_*.md` and `docs/codex_questions.md`
  for open questions and blockers.
- Run `git log --oneline -15` and `git status` to see what just happened and
  whether the tree is clean.

### 2. State a plan
Before editing, write down (in your reply, or a scratch note) what you intend to
change and why. If the task is ambiguous or could go several ways, **ask first**
rather than guessing.

### 3. Make the change
- Keep changes focused. One logical change per commit.
- Match the existing style. Don't reformat unrelated code.
- If you touch precision configs, see the **hls4ml gotcha** below — the config
  is duplicated in four files and they must stay in sync.

### 4. Verify what you can
- At minimum, syntax-check edited Python (`python -c "import ast; ..."`) and
  `bash -n` edited shell scripts.
- If you can't run the real pipeline (no GPU, no Vivado, no model file), **say
  so explicitly** and mark the change as unverified. Never imply something
  passed when you didn't run it.

### 5. Update the docs that the change affects
- Bug fix or debugging insight → add to the relevant `docs/*.md` diagnosis log
  (e.g. `docs/hls4ml_precision_bugs.md`), as a new dated step.
- Changed behavior, setup, or results → update `README.md` and/or
  `hls4ml/README.md`.
- New numbers (correlation, AUC, latency, resources) → record them where the
  old numbers live; don't leave stale figures.

### 6. Write patch notes — MANDATORY
Append an entry to **`docs/CHANGELOG.md`** for every session that changes the
repo. Format:

```markdown
## YYYY-MM-DD — short title

**Who:** human name / LLM name (e.g. "Codex", "Claude")
**Commits:** <short shas>

### What changed
- bullet points, plain language

### Why
- the reason / the bug / the request

### Verified
- what you actually ran and what the result was
- explicitly list what you could NOT verify and why

### Still broken / unfinished
- anything left undone, with enough detail for the next person to pick up
```

If `docs/CHANGELOG.md` doesn't exist yet, create it with this entry as the first
one.

### 7. Update "what's next" — MANDATORY
Update `docs/NEXT_STEPS.md` so it reflects reality *after* your change:
- Cross off / remove what you completed.
- Add anything new you discovered that needs doing.
- Re-order so the top item is genuinely the next thing to do.
- If you left a blocker, it goes here AND in the changelog "still broken"
  section.

**An LLM that does not do steps 6 and 7 has not finished the task.**

### 8. Commit and (if asked) push
See the version-control section below.

---

## Version control rules

### Commits
- **Use Conventional Commits.** Prefix every commit with a type:
  - `feat:` new capability
  - `fix:` bug fix
  - `docs:` documentation only
  - `chore:` tooling, gitignore, file moves, cleanup
  - `refactor:` behavior-preserving restructure
  - `test:` tests only
- Subject line: imperative, ≤ ~72 chars, no trailing period.
  Good: `fix(hls4ml): correct input_norm 2x amplification`
- **Write a body** for anything non-trivial. Explain the *why* and the *root
  cause*, not just the *what* — the diff already shows the what. Wrap at ~72.
- **One logical change per commit.** Don't bundle a bug fix with a refactor and
  a doc rewrite. Separate commits are easier to review and revert.
- Reference the relevant doc (`See docs/hls4ml_precision_bugs.md`) when a commit
  implements something documented there.

### Branches
- `main` is the source of truth and should always be in a runnable / coherent
  state.
- For anything risky, multi-commit, or that the owner should review, use a
  branch: `fix/...`, `feat/...`, `docs/...`, then open a PR. Don't push
  half-broken work straight to `main`.
- Small, safe, self-contained changes may go to `main` directly **only with the
  owner's go-ahead.** When in doubt, branch.

### Pushing — ask first
- **An LLM must get explicit confirmation before pushing to a remote**, unless
  the user already said "push it" for this task. Committing locally is fine;
  pushing is a side effect the owner should approve.
- Never force-push `main`. Never rewrite published history without explicit
  permission.

### What never gets committed
- Large binaries: model weights (`*.h5`, `*.onnx`), data (`*.root`, `*.npy`),
  generated HLS projects (`models/hls4ml_*`), plots (`*.png`, `*.pdf` outside
  `docs/`), `wandb/`, logs. These are already in `.gitignore` — keep it that
  way. If something large needs to be shareable, document *where it lives* (see
  `models/MODEL.md`), don't commit the blob.
- Secrets / tokens of any kind. Ever.
- If you find a committed binary or secret, flag it and propose removing it in a
  `chore:` commit.

### Keeping the tree clean
- Run `git status` before committing; commit intentionally, not `git add -A`
  without looking. Know what's in your commit.
- Leave the working tree clean at the end of a session (everything either
  committed or explicitly noted as WIP in the changelog).

---

## Repo-specific gotchas (don't relearn these the hard way)

- **The hls4ml precision config is duplicated in FOUR files:**
  `hls4ml/hls_convert_v2.py`, `hls4ml/hls_convert_iostream.py`,
  `hls4ml/hls_trace.py`, and `hls4ml/hls_build.py`. If you change a LayerNorm
  or dense precision, change it in **all four** or synthesis/trace will silently
  disagree with conversion. (A real bug from 2026-05-31: the fix landed in the
  convert script but not `hls_build.py`, which would have baked the bug into
  firmware.) Refactoring these into one shared module is a tracked nice-to-have.
- **`hls_model.compile()` / `.write()` regenerates firmware** — never hand-edit
  generated `defines.h` / `parameters.h`. Fixes go in the Python config or the
  hls4ml source patches under `patches/hls4ml/`.
- **A fresh clone can't run the pipeline until** you run
  `bash hls4ml/setup_hls4ml.sh` (installs patched hls4ml) and place the model at
  `models/deepsets_d64_l3_ffn128/deepsets_clean.h5` (see `models/MODEL.md`).
- **Don't claim synthesis/accuracy numbers you didn't measure.** If a result
  isn't from a run you actually did, mark it as a target or a placeholder.

---

## For LLMs specifically — a short checklist to self-verify before you stop

- [ ] I read `RULES.md`, `README.md`, and `docs/NEXT_STEPS.md` at the start.
- [ ] My code change is focused and style-matching.
- [ ] I syntax-checked / `bash -n`'d what I edited.
- [ ] I updated every doc the change affects, with no stale numbers.
- [ ] I added a dated entry to `docs/CHANGELOG.md` (what / why / verified /
      still-broken).
- [ ] I updated `docs/NEXT_STEPS.md` to reflect reality after my change.
- [ ] I used a Conventional Commit message with a real body.
- [ ] I did NOT push without permission.
- [ ] I clearly told the user what I could not verify and what's next.

If any box is unchecked, you are not done.
