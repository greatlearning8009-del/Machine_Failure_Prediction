# Kept for anyone who wants to containerise the app for a non-Codespaces host.
# Codespaces itself does NOT use this Dockerfile — it uses .devcontainer/devcontainer.json.
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false"]
