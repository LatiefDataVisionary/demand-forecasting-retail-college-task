# # Use a lightweight Python base image
# # Ganti python:3.9-slim-buster menjadi:
# FROM python:3.10-slim-buster

# # Set the working directory inside the container
# WORKDIR /app

# # Copy the requirements file into the working directory
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the 'assets' and 'code' directories into the working directory
# COPY assets ./assets 
# COPY code ./code

# # Expose the port that the Flask app will run on
# EXPOSE 5000

# # Command to run the Flask application using Gunicorn
# # The 'app' refers to the Flask instance in app.py
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "code.app:app"]

FROM python:3.11-slim-bullseye

WORKDIR /app

# Install OS dependencies required for LightGBM (libgomp1 for OpenMP runtime)
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY assets ./assets
COPY code ./code

# PENTING: Pindahkan WORKDIR ke dalam folder code
WORKDIR /app/code

EXPOSE 5000

# File app.py berada di /app/code, jadi kita tambahkan /app/code ke PYTHONPATH
ENV PYTHONPATH /app/code:$PYTHONPATH

# Perintah untuk menjalankan Flask application menggunakan Gunicorn
# Sekarang kita bisa merujuk langsung ke nama file 'app'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]