# Development Workflow

## Typical change loop

1. Create branch
2. Implement code changes
3. Update docs if behavior/API/CLI changed
4. Run focused tests and docs build
5. Open PR

## Docs expectations

If you change user-visible behavior, update one or more of:

- how-to guides
- reference docs
- architecture notes
- known inconsistencies page

## Suggested local checks

```bash
make docs
make docs-linkcheck
poetry run pytest tests/backend -q
```
