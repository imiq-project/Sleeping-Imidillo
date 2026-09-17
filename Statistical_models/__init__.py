"""
Deterministic statistics for the monthly report.

Each module turns raw sensor series into a small table of facts: every number
carries a confidence interval, a p-value and the data coverage behind it. The
LLM writer may only cite numbers from these tables and the verifier checks its
prose against them; neither ever sees a raw series.

Modules are runnable on their own (`python -m Statistical_models.<name>`) and write
their facts JSON to output/<month>/ (gitignored).
"""
