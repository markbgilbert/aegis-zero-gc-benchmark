# Aegis Zero-GC Flat Arena Benchmark Container
# Ubuntu 24.04 LTS with GCC, CMake, Node.js, and Python verification tools

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gcc \
        clang \
        git \
        nodejs \
        python3 \
        python3-pip \
        python3-venv \
        libjemalloc2 \
        libjemalloc-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/aegis-zero-gc-benchmark

COPY . .

# Build native C benchmark via CMake
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release

# Compile shared kernels
RUN gcc -O3 -mavx2 -shared -fPIC aegis_feeder.c -o aegis_feeder.so && \
    gcc -O3 -mavx2 -shared -fPIC aegis_telemetry_engine.c -o aegis_telemetry.so

# Default execution: Run 1B native benchmark and cross-language JS verification
CMD ["/bin/bash", "-c", "./build/bench_1b 100000000 && node bench_1b.js"]
