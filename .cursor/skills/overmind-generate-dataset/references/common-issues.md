# Common Issues

| Problem                                              | Fix                                                                                       |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `generate_diverse_synthetic_data` not found          | overmind is not installed — run `pip install overmind`, then re-run                       |
| Model auth error                                     | Check `.overmind/.env` or `.env` for `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`               |
| 0 cases generated                                    | Increase temperature or reduce `NUM_SAMPLES` per run; retry                               |
| Input schema missing fields                          | Re-read the entrypoint and check `*args`/`**kwargs` usage                                 |
| >20% cases dropped by schema filter                  | Tighten the `eval_spec` and regenerate; the LLM is producing inconsistent keys            |
| Smoke test: `TypeError: unexpected keyword argument` | The detected `input_schema` has extra or wrong parameter names — fix and regenerate       |
| Smoke test: API / auth errors from the agent         | Expected if the agent calls external APIs; mock them or ignore and focus on schema errors |
