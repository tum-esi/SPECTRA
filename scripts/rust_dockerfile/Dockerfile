# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.8.2-devel-ubuntu24.04

ARG RUST_VERSION=1.97.1

WORKDIR /workspace

# The CUDA development image supplies nvcc. Install the remaining native,
# Python, and Rust tooling used by the project.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/opt/venv/bin:/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain "${RUST_VERSION}" \
    && rm /tmp/rustup-init.sh

RUN python3 -m venv /opt/venv

COPY scripts/python/requirements.txt /tmp/python-requirements.txt
RUN pip install --no-cache-dir -r /tmp/python-requirements.txt

RUN git config --system --add safe.directory /workspace

CMD ["bash"]

