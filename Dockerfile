FROM rayproject/ray:1.12.0 as base_image
RUN sudo apt-get update
# for reading flac files
RUN sudo apt install libsndfile1-dev -y
# for netstat
RUN sudo apt-get install net-tools
RUN pip install torch
RUN pip install transformers
RUN sudo apt-get install nano
FROM base_image AS builder
ENV BASE_DIR=/app
RUN sudo mkdir -p $BASE_DIR
WORKDIR $BASE_DIR
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# for spacy
RUN python -m spacy download en_core_web_sm
COPY driver.py config.env 1284-1181-0004.flac ./
COPY src src/
RUN sudo mkdir -p ~/.cache
RUN sudo chown ray ~/.cache
# CMD ./run.sh
