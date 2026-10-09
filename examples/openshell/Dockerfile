# Workload image for the demo sandbox: Python stdlib only, non-root.
FROM python:3.12-slim
RUN useradd -m -u 1000 agent
COPY agent.py /app/agent.py
USER agent
WORKDIR /home/agent
