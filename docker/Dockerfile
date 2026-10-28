FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libegl1 \
        libgl1 \
        libgl1-mesa-dri \
        libglfw3 \
        libglib2.0-0 \
        libglx-mesa0 \
        libosmesa6 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt \
    && python3 -m pip check

COPY fruitfly_brain/ ./fruitfly_brain/
COPY tests/ ./tests/
COPY data/ /data/

# Preserve the CLI's project-relative data defaults while storing data at /data.
RUN ln -s /data /app/data

ENTRYPOINT ["python3", "-m", "fruitfly_brain.closed_loop"]
CMD ["--duration", "1.0"]