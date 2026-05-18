# Arc AGI 2 — Claude rules

## Commit & push shorthand

When the user says **"push"** (alone or as the main intent of a message):
1. Stage all changed and untracked files relevant to recent work (never `.env` or secrets).
2. Write a concise commit message describing what changed and why.
3. Commit and push to the current branch without asking for confirmation.
