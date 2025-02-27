# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY ./src/simple_api.py simple_api.py
COPY ../../../models/best_random_forests.joblib best_random_forests.joblib
# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn pandas joblib

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the application with Uvicorn
CMD ["uvicorn", "simple_api:app", "--host", "0.0.0.0", "--port", "8000"]