# decide — typed decision engine CLI (multi-arch: linux/amd64, linux/arm64)
#
# Build (example: arm64 for Raspberry Pi):
#   podman build --platform linux/arm64 --build-arg TARGETARCH=arm64 \
#     -t wallydk24/decide:arm64 .
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
#
# deps/ holds pre-unpacked `requests` trees per arch (amd64/arm64) so the
# build needs no network and no emulation: COPY is arch-independent.

FROM python:3.12-slim
ARG TARGETARCH=amd64
COPY deps/${TARGETARCH}/ /usr/local/lib/python3.12/site-packages/

WORKDIR /app
COPY decide.py sets.yml ./
ENV HOME=/tmp
USER 1000

ENTRYPOINT ["python3", "/app/decide.py"]
CMD ["--help"]
