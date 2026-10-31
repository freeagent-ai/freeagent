# Rules for AI assistants working in this repo

Project: FreeAgent ("Buying Points, Not Names"), a university AIML + Maths capstone.
The architecture and maths are designed by the team with Claude; your job is to RUN,
DEBUG and BUILD UI/plots exactly as instructed.

1. Do NOT change any formula, model or function signature in `src/freeagent/` unless
   the prompt explicitly says to. If something in `src/` errors, report the full
   traceback instead of rewriting the maths.
2. Never invent numbers (accuracies, fees, results). Every number must come from running code.
3. Never edit a test to make it pass. If a test fails, report it.
4. Never use data from after a prediction cut-off date when training (no future leakage).
5. Run `python tests/run_tests.py` after any change; all tests must pass before committing.
6. Raw data stays in `data/` and is never committed to git.
7. Keep changes small and explain each one in plain English.
