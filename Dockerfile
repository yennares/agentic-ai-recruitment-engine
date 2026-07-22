FROM python:3.9-slim
# Set the working directory in the container
WORKDIR /app
# Copy the requirements file into the container at /app
COPY requirements.txt .
# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the current directory contents into the container at /app
COPY . .
# Expose the port that your Streamlit app will run on
EXPOSE 8080
# Command to run the Streamlit app
# CMD ["python", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
CMD ["python", "app.py"]