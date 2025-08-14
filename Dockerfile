# Use official Python 3.13 base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install mlflow 3.x
RUN pip install mlflow==3.*

# Expose port 5000
EXPOSE 5000

# Command to run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]