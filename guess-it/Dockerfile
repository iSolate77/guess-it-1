# Start with the Node.js base image
FROM node:16 as node-base

WORKDIR /usr/src/app

# Install system dependencies and Python3
RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    gfortran \
    python3 \
    python3-pip \
    python3-venv \
    curl

# Install Node.js
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get install -y nodejs

# Create a Python virtual environment and activate it
RUN python3 -m venv /usr/src/app/venv
ENV PATH="/usr/src/app/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install collections-extended
RUN pip install numpy scikit-learn

# Set up Node.js application
COPY package*.json ./
RUN npm install
COPY . .

EXPOSE 3000

CMD [ "node", "server.js" ]
