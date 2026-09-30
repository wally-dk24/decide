# decide — typed decision engine CLI
#
# Build:  docker build -t wallydk24/decide .
# Try:    docker run --rm wallydk24/decide --help
# Decide: docker run --rm -e DECIDE_LIQUID_API_KEY=$KEY wallydk24/decide \
#           --set spam_check --question "Is this spam?" --input /data/msg.txt
#
# Keys are NEVER baked into the image — pass them at runtime via env vars:
#   DECIDE_LIQUID_API_KEY  (Liquid d1 decision model)
#   DECIDE_LLM_URL / DECIDE_LLM_KEY / DECIDE_LLM_MODEL  (OpenAI-compatible fallback)
#
# The bundled sets.yml ships as default outcome sets; override with:
#   -v ./my-sets.yml:/app/sets.yml  or  --outcomes /path/inside/container

FROM python:3.12-slim

# requests is vendored as wheels (see wheels/) so the build needs no network.
COPY wheels/ /wheels/
RUN pip install --no-cache-dir --no-index /wheels/*.whl && rm -rf /wheels

WORKDIR /app
COPY decide.py sets.yml ./
RUN useradd -m decide && chown -R decide:decide /app
USER decide

ENTRYPOINT ["python3", "/app/decide.py"]
CMD ["--help"]
