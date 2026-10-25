# Reproducible environment for building and verifying kickmix circuits.
# Always built for linux/amd64 (see docker-compose.yml) so that the Kaggle-shipped
# data/kickmix/kickmix.so (x86_64) loads and simulation matches the leaderboard image.
FROM --platform=linux/amd64 python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential g++ git ca-certificates zip unzip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Python deps for our own tooling and the verifiers. numpy is kickmix's only runtime dep.
RUN pip install numpy pytest kaggle

# Build kickmix from the pinned Qualtran submodule (pybind11 extension, C++20, see kickmix/setup.py).
# Copied separately so the long compile is cached independently of our own sources.
COPY third_party/Qualtran/kickmix /opt/kickmix-src
RUN pip install /opt/kickmix-src

# The dfthcsos verifier also needs scipy (linear_sum_assignment); separate layer so kickmix stays cached.
RUN pip install scipy

# Our repo is bind-mounted at /work by docker-compose; nothing else is baked in.
# To use the Kaggle-provided binary instead of the source build:
#   PYTHONPATH=/work/data/kickmix python ...
CMD ["bash"]
