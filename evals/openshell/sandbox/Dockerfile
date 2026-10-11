# Workload image for the benchmark sandbox: Python stdlib only, non-root.
# The build context is assembled by run_bench.py (this file, bench_agent.py and the generated plan.json).
FROM python:3.12-slim
RUN useradd -m -u 1000 agent
COPY bench_agent.py plan.json /app/
USER agent
WORKDIR /home/agent
