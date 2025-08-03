# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port for Gradio (default is 7860, but can be changed)
EXPOSE 7860


# Command to run your application
# Assuming your main script is app.py and uses Gradio for UI.
# If it uses Fire or another method, update this CMD accordingly.
CMD ["python", "app.py"]  # Example: If app.py contains gradio.launch() or similar