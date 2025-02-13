# Based on NVIDIA Isaac Sim official image
FROM nvcr.io/nvidia/isaac-sim:4.0.0

# Set non-interactive environment (avoid installation prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Set default working directory
WORKDIR /isaac-sim

# Update packages and install common tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    vim \
    htop \
    cmake \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh && \
    bash /root/miniconda.sh -b -p /root/miniconda && \
    rm /root/miniconda.sh

# Add conda to PATH
ENV PATH="/root/miniconda/bin:${PATH}"

# Initialize conda
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc

# Create and configure Python 3.10 conda environment
RUN conda create -n py10 python=3.10 -y && \
    conda clean -a -y 

SHELL ["conda", "run", "-n", "py10", "/bin/bash", "-c"]

WORKDIR /root

# Clone IsaacLab repository to /root/IsaacLab
RUN wget https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v1.1.0.zip -O /root/IsaacLab.zip && \
    unzip /root/IsaacLab.zip && \
    mv IsaacLab-1.1.0 IsaacLab && \
    rm /root/IsaacLab.zip && \
    ln -s /isaac-sim /root/IsaacLab/_isaac_sim

WORKDIR /root/IsaacLab

# Set Isaac Sim related environment variables
ENV ISAACSIM_PATH="/isaac-sim"
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

RUN ./isaaclab.sh --conda py10

RUN ./isaaclab.sh --install 

# Install other dependencies
# gaussian-splatting
RUN conda install -c conda-forge colmap -y
RUN mkdir /root/dev
RUN mkdir -p /root/dev/real2sim2real
WORKDIR /root/dev/real2sim2real
COPY . .

# Install real2sim2real
WORKDIR /root/dev/real2sim2real
RUN cp setup.py /root/dev
RUN pip install -e ..
RUN pip install -r requirements.txt

WORKDIR /isaac-sim/
RUN ln -s /root/dev/real2sim2real /isaac-sim/src
