# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the API
CMD ["python", "api.py"]
