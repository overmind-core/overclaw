# Repair Mode — Fixing Existing Broken Artifacts

When the user points the skill at an agent that already has a `setup_spec/` directory:

1. Read `eval_spec.json` and `policies.md`.
1. Run static analysis on the agent (Step 1 of the main workflow).
1. Diff: list every field that is wrong (collapsed input, missing output keys, weight sum ≠ 100, empty policy lists, mismatched enum values vs code).
1. Show the diff to the user. `AskQuestion`: *"Apply all fixes / pick which to apply / abort"*.
1. Apply selected fixes, re-run validation, re-save.

The diff format must be concrete — show the *current* vs *proposed* value side by side, not a vague "this looks wrong".
