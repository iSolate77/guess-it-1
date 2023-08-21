# Use the official Python 3.8 image as the base image
FROM python:3.8-slim

# Install system dependencies for numpy and Node.js
RUN apt-get update && apt-get install -y \
    curl \
    libblas-dev \
    liblapack-dev \
    gfortran \
    gnupg2

# Install Node.js
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs

# Check node and npm
RUN node -v
RUN npm -v

# Install numpy and scikit-learn
RUN pip install numpy scikit-learn

# Set the working directory
WORKDIR /usr/src/app

# Install app dependencies
COPY package*.json ./
RUN npm install

# Bundle app source
COPY . .

# Clean up
RUN apt-get purge -y --auto-remove curl && rm -rf /var/lib/apt/lists/*

# Expose the port the app runs on
EXPOSE 3000

# Command to run the application
CMD [ "node", "server.js" ]

