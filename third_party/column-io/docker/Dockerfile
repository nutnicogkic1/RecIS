ARG BASE_XREC_IMAGE
from ${BASE_XREC_IMAGE}

COPY . /workspace/column-io/

RUN cd /workspace/column-io/ && sh tools/build_and_install.sh 0
RUN cd / && rm -rf /workspace/column-io/
