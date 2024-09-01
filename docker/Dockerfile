FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev libgmp-dev cmake \
libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev \
libffi-dev libdb-dev ca-certificates rustc git graphviz

# Install Rust and Cargo
RUN apt-get update && \
    apt-get install -y curl && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Cargo to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

RUN wget --no-check-certificate https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz \
&& tar -xf Python-3.10.0.tgz \
&& cd Python-3.10.0 \
&& ./configure --enable-optimizations\
&& make \
&& make install

RUN apt-get autoremove -y

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir ray toml loguru tqdm torch==2.0.0 openai pytorch_lightning torchmetrics transformers deepspeed pytest lxml scikit-learn matplotlib optuna

ENV ELAN_HOME="/.elan"
ENV PATH="${ELAN_HOME}/bin:${PATH}"
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y

RUN ls -la /.elan || echo '.elan directory does not exist'

RUN chmod -R a+w /.elan || echo "Failed to change permissions of /.elan"

ARG USER_ID

ARG GROUP_ID
ARG USERNAME

RUN echo "USER_ID: ${USER_ID}, GROUP_ID: ${GROUP_ID}, USERNAME: ${USERNAME}"

ENV USER_ID ${USER_ID}
ENV GROUP_ID ${GROUP_ID}
ENV USERNAME ${USERNAME}

RUN groupadd -g ${GROUP_ID} ${USERNAME}
RUN useradd -u ${USER_ID} -g ${USERNAME} -m ${USERNAME}
USER ${USERNAME}

WORKDIR /home/myuser/src
RUN chmod -R a+w /home/myuser/src
ENV PYTHONPATH="${PYTHONPATH}:/home/myuser/src"
