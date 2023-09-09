# Use the official Rust image as a base
FROM rust:1.57

# Set up the working directory in the container
WORKDIR /usr/src/myrustapp

# Copy the Rust project into the container
COPY . .

# Build the Rust project
RUN cargo build --release
