# Use an official Python runtime (3.10-slim) as the base image
FROM python:3.10-slim

# Update package lists and install system dependencies, including distutils and gcc
RUN apt-get update && apt-get install -y python3-distutils gcc

# Set the working directory to /app
WORKDIR /app

# Copy the entire repository into /app
COPY . /app

# Upgrade pip and install Python dependencies from requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the Flask app
EXPOSE 5000

# Set an environment variable for production
ENV FLASK_ENV=production

# Command to run the Flask app
CMD ["python", "flask_app.py"]
