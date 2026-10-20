# Release-guard reference outputs

`guard_verdicts.json` records what the release guard (`scripts/release_guard/check_release.py`) reports on a
fixed synthetic corpus. It is read by `scripts/release_guard/tests/test_release_guard.py`, which checks that the
guard keeps every rule and verdict.

What is recorded, from the `check_release.py` module (no git calls):

- `rules`: each rule's id, scope, level, secret flag, regex flags and the sha256 of its pattern;
- `tables`: FORBIDDEN_GLOBS, FORBIDDEN_EXT, TEXT_EXT and the size limits;
- `scan_cases`: `scan_text` findings for synthetic lines, once with every rule (`full`) and once with the secret
  tier only (`secret_tier`), including a known-secret canary and harvested tokens;
- `extra_denylist`: the rules `load_extra` builds from a small denylist file;
- `path_cases`: `structural` findings (name-based rules) for synthetic paths;
- `allow_cases`: `allowed()` lookups for synthetic paths.

All inputs are synthetic (fake tokens drawn from `random.Random(20261009)`, `example.com` hosts, made-up paths)
and stored base64-encoded, so this file never trips the guard or gitleaks itself.

Differences asserted by the test on top of the recorded verdicts: `.pyc` and `.pyo` are forbidden extensions;
FORBIDDEN_GLOBS also match under `opsd/` (the package); the guard's earliest location
`opsd/tools/release_guard/` is exempt in history mode only.
