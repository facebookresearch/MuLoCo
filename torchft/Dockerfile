FROM ghcr.io/pyo3/maturin

RUN PB_REL="https://github.com/protocolbuffers/protobuf/releases" && curl -LO $PB_REL/download/v25.1/protoc-25.1-linux-x86_64.zip
RUN unzip protoc-25.1-linux-x86_64.zip -d /usr/

#ENTRYPOINT ["/bin/bash"]