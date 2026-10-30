---
id: NN-short-name
title: One line describing the result
depends_on: []
estimate: N days
real_weights: none | stage-b | required
---

## Goal

What exists after this run and why it matters for the port.

## Read first

- `diffusers/<path>:<lines>` (0.42.1) — what to look for there.
- Earlier modules this spec builds on.

## Deliverables

- `src/kandinsky6_mlx/...` — module and public API.
- `tests/...` — test files.
- `reports/NN-short-name.json`

## Implementation notes

Facts about the reference that are easy to get wrong: layouts, dtypes, ordering, shapes.

## Tests

Each test with the property it proves.

## Validation runs

Exact commands and what numbers to record.

## Acceptance criteria

- [ ] Checkable criterion with a command or metric.

## Out of scope

What not to do in this run.

## Stop conditions

When to stop and report `BLOCKED` or `FAIL` instead of improvising.
