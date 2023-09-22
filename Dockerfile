FROM python:3.9
RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
        zip build-essential ffmpeg libsm6 libxext6 wget less vim && \
     rm -rf /var/lib/{apt,dpkg,cache,log}/

ARG UNAME
ARG GID
ARG UID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && pip install setuptools==65.5.0
RUN pip install -r requirements.txt
ENV PYTHONPATH "$PYTHONPATH:/workspace"
ENV PATH "$PATH:/home/"$UNAME"/.local/bin"
ENV OMP_NUM_THREADS=1