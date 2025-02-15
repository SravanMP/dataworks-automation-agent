import os
import requests
import subprocess
import sqlite3
import duckdb
import markdown
import pandas as pd
from fastapi import FastAPI, HTTPException

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Fetch AIPROXY_TOKEN from environment variable
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN not set. Please configure it in your environment.")

app = FastAPI()

# B1: Restrict data access outside /data
def B1(filepath):
    """Ensure the file path starts with /data to prevent unauthorized access."""
    if not filepath.startswith('/data'):
        logging.error(f"Unauthorized access attempt: {filepath}")
        raise HTTPException(status_code=403, detail="Access outside /data is not allowed.")
    return True

# B2: Prevent data deletion
def B2(filepath, mode='r'):
    """Ensure that files are only opened in read mode and not deleted."""
    if 'w' in mode or 'x' in mode or 'a' in mode:
        logging.error(f"Unauthorized modification attempt: {filepath}")
        raise PermissionError("Modifications to files are not allowed.")
    return True

# B3: Fetch Data from an API and Save It
def B3(url, save_path):
    if not B1(save_path):
        raise HTTPException(status_code=403, detail="Access outside /data is not allowed.")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'w') as file:
            file.write(response.text)
        logging.info(f"Data fetched from {url} and saved to {save_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to fetch data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# B4: Clone a Git Repo and Make a Commit
def B4(repo_url, commit_message):
    repo_path = "/data/repo"
    if not B1(repo_path):
        raise HTTPException(status_code=403, detail="Access outside /data is not allowed.")
    
    try:
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
        subprocess.run(["git", "-C", repo_path, "commit", "-m", commit_message], check=True)
        logging.info(f"Repo cloned and commit made: {repo_url}")
        return True
    except Exception as e:
        logging.error(f"Failed to clone repo or make commit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# B5: Run a SQL Query on SQLite or DuckDB
def B5(db_path, query, output_filename):
    if not B1(db_path) or not B1(output_filename):
        raise HTTPException(status_code=403, detail="Access outside /data is not allowed.")

    try:
        db_type = 'sqlite' if db_path.endswith('.db') else 'duckdb'
        conn = sqlite3.connect(db_path) if db_type == 'sqlite' else duckdb.connect(db_path)
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
        conn.close()

        with open(output_filename, 'w') as file:
            file.write(str(result))
        
        logging.info(f"Query executed and result saved to {output_filename}")
        return True
    except Exception as e:
        logging.error(f"Failed to execute query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# B6: Web Scraping
def B6(url, output_filename):
    if not B1(output_filename):
        raise HTTPException(status_code=403, detail="Access outside /data is not allowed.")

    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_filename, 'w') as file:
            file.write(response.text)
        logging.info(f"Website scraped and saved to {output_filename}")
        return True
    except Exception as e:
        logging.error(f"Failed to scrape website: {e}")
        raise HTTPException(status_code=500, detail=str(e))

'''
# B7: Image Processing (Compression or Resizing)
def B7(image_path, output_path, resize=None):
    if not B1(image_path) or not B1(output_path):
        raise HTTPException(status_code=403, detail="Access outside /data is not allowed.")

    try:
        img = Image.open(image_path)
        if resize:
            img = img.resize(resize)
        img.save(output_path)
        logging.info(f"Image processed and saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to process image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''

# B8: Audio Transcription (Dummy Implementation)
def B8(audio_path):
    if not B1(audio_path):
        raise HTTPException(status_code=403, detail="Access outside /data is not allowed.")

    try:
        # Integrate with an external API like OpenAI Whisper here
        transcription = "Transcription feature needs integration with an external API like Whisper."
        logging.info(f"Audio transcription requested for {audio_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to transcribe audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# B9: Convert Markdown to HTML
def B9(md_path, output_path):
    if not B1(md_path) or not B1(output_path):
        raise HTTPException(status_code=403, detail="Access outside /data is not allowed.")

    try:
        with open(md_path, 'r') as file:
            html = markdown.markdown(file.read())
        with open(output_path, 'w') as file:
            file.write(html)
        logging.info(f"Markdown converted and saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to convert markdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# B10: API Endpoint for Filtering CSV and Returning JSON Data
def B10(csv_path, filter_column, filter_value):
    if not B1(csv_path):
        raise HTTPException(status_code=403, detail="Access outside /data is not allowed.")

    try:
        df = pd.read_csv(csv_path)
        filtered = df[df[filter_column] == filter_value]
        logging.info(f"CSV filtered and returned JSON data")
        return True
    except Exception as e:
        logging.error(f"Failed to filter CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))