import os
import requests
import subprocess
import sqlite3
import duckdb
import markdown
import pandas as pd
from fastapi import FastAPI, HTTPException
import json
import datetime
import re
import whisper
from PIL import Image
from typing import Optional, Tuple
from pathlib import Path


# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Fetch AIPROXY_TOKEN from environment variable
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN not set. Please configure it in your environment.")

app = FastAPI()

# B1: Restrict data access outside /data
def B1(filepath, base_dir="/data"):
    """Check if a file is inside the /data directory."""
    abs_filepath = os.path.abspath(filepath)
    abs_base_dir = os.path.abspath(base_dir)
    if not abs_filepath.startswith(abs_base_dir):
        raise HTTPException(status_code=403, detail=f"Access to this file: {filepath} is forbidden")
    else:
        return True

# B2: Prevent data deletion
def B2(filepath, mode='r'):
    """Ensure that files are only opened in read mode and not deleted."""
    if 'w' in mode or 'x' in mode or 'a' in mode:
        logging.error(f"Unauthorized modification attempt: {filepath}")
        raise PermissionError("Modifications to files are not allowed.")
    return True

# B3: Fetch Data from an API and Save It
def B3(filename, targetfile):
    try:
        # Make the API request
        response = requests.get(filename)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        # Parse JSON response
        data = response.json()

        # Save the data to a file
        with open(targetfile, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        print(f"Data successfully saved to {targetfile}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

# B4: Clone a Git Repo and Make a Commit

def run_command(command: str, cwd: Optional[str] = None) -> Tuple[bool, str]:
    """Execute a shell command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)

def validate_git_url(url: str) -> bool:
    """Validate if the provided URL is a valid git repository URL."""
    git_url_pattern = r'^(https?:\/\/)?([\w\d\-_]+@)?([a-zA-Z\d\-_]+\.[a-zA-Z\d\-_]+)(:\d+)?\/.*?\.git$'
    return bool(re.match(git_url_pattern, url))

def setup_git_config(name: str, email: str) -> bool:
    """Setup git configuration."""
    success, _ = run_command(f'git config --global user.name "{name}"')
    if not success:
        return False
    success, _ = run_command(f'git config --global user.email "{email}"')
    return success

def handle_git_repo(
    repo_url: str,
    file_content: str,
    commit_message: str,
    file_path: str = "README.md",
    branch: str = "main",
    git_token: Optional[str] = None,
    work_dir: str = "data"
) -> Tuple[bool, str]:
    """
    Clone a repository, make changes, and commit them.
    """
    if not validate_git_url(repo_url):
        return False, "Invalid git repository URL"

    # Extract repo name from URL
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join(work_dir, repo_name)

    # Store current directory
    original_dir = os.getcwd()
    
    try:
        # Construct URL with auth token if provided
        if git_token:
            auth_url = repo_url.replace('https://', f'https://oauth2:{git_token}@')
        else:
            auth_url = repo_url

        # Clone or pull repository
        if os.path.exists(repo_path):
            success, output = run_command(f"git pull origin {branch}", repo_path)
            if not success:
                return False, f"Failed to pull repository: {output}"
        else:
            success, output = run_command(f"git clone {auth_url}", work_dir)
            if not success:
                return False, f"Failed to clone repository: {output}"

        # Change to repo directory
        os.chdir(repo_path)

        # Create/update file
        file_full_path = os.path.join(repo_path, file_path)
        os.makedirs(os.path.dirname(file_full_path), exist_ok=True)
        with open(file_full_path, 'w') as f:
            f.write(file_content)

        # Stage changes
        success, output = run_command("git add .")
        if not success:
            return False, f"Failed to stage changes: {output}"

        # Commit changes
        success, output = run_command(f'git commit -m "{commit_message}"')
        if not success:
            return False, f"Failed to commit changes: {output}"

        # Push changes
        success, output = run_command(f"git push origin {branch}")
        if not success:
            return False, f"Failed to push changes: {output}"

        return True, "Successfully cloned repository and committed changes"

    except Exception as e:
        return False, f"An error occurred: {str(e)}"
    finally:
        os.chdir(original_dir)

def B4(task_description: str) -> str:
    """
    Handle git repository operations (Task B4).
    
    Args:
        task_description: Description of the git task to perform
        
    Returns:
        String describing the result of the operation
    """
    try:
        # Get git credentials from environment variables
        git_token = os.getenv('GITHUB_TOKEN')
        git_name = os.getenv('GIT_NAME', 'Automated User')
        git_email = os.getenv('GIT_EMAIL', 'automated@example.com')
        
        # Setup git configuration
        if not setup_git_config(git_name, git_email):
            return "Failed to setup git configuration"

        # Extract repository URL from task description
        # In practice, you would use an LLM to parse this information
        repo_url = task_description.split('repo:')[-1].strip() if 'repo:' in task_description else ''
        if not repo_url:
            return "No repository URL found in task description"

        # Create sample content (in practice, this would come from task description)
        file_content = f"""# Automated Update
        
Updated at: {datetime.datetime.now().isoformat()}
This is an automated commit based on the task: {task_description}
"""

        # Handle repository operations
        success, message = handle_git_repo(
            repo_url=repo_url,
            file_content=file_content,
            commit_message=f"Automated update: {datetime.datetime.now().isoformat()}",
            git_token=git_token
        )

        return message

    except Exception as e:
        return f"Failed to process git task: {str(e)}"

# B5: Run a SQL Query on SQLite or DuckDB
def B5(db_path, targetfile, query):
    print(f"Running SQL query: {query}, {db_path}, {targetfile}")
    """
    Runs a SQL query on a SQLite or DuckDB database and saves the result.
    
    Parameters:
        db_path (str): Path to the SQLite (.db) or DuckDB (.duckdb) database file.
        query (str): SQL query to execute.
        output_file (str): File to save the results.
        output_format (str): "csv" or "json" (default: "csv").
    """
    # Determine database type (SQLite or DuckDB)
    is_duckdb = db_path.endswith(".duckdb")
    
    if not targetfile:
        targetfile = "./data/output_B5.csv"
    elif targetfile.startswith("/"):
        targetfile = f".{targetfile}"
    
    output_format = "csv" if targetfile.endswith(".csv") else "json" if targetfile.endswith(".json") else "txt"

    # Connect to the database
    conn = duckdb.connect(db_path) if is_duckdb else sqlite3.connect(db_path)
    
    try:
        # Execute the query and fetch results into a DataFrame
        df = pd.read_sql_query(query, conn)

        # Save results
        if output_format == "json":
            df.to_json(targetfile, orient="records", indent=4)
        elif output_format == "txt":
            df.to_csv(targetfile, sep="\t", index=False)
        else:  # Default is CSV
            df.to_csv(targetfile, index=False)

        print(f"✅ Query executed successfully. Results saved to {targetfile}")
        
        return targetfile
    except Exception as e:
        print(f"❌ Error executing query: {e}")
    finally:
        conn.close()

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

def B7(input_path, output_path, resize_width=None, resize_height=None, quality=80):
    print(f"Processing image: {input_path}, {output_path}, {resize_width}, {resize_height}, {quality}")
    
    if not os.path.exists(input_path) or not output_path or not (resize_width or resize_height or quality):
        raise HTTPException(status_code=400, detail=f"Invalid input parameters: input_path: {input_path}, output_path: {output_path} and one of (resize_width : {resize_width}, resize_height: {resize_height}, quality: {quality})")
    
    """
    Compress or resize an image and save it to output_path. credit_card.png
    
    Parameters:
        input_path (str): Path to the original image.
        output_path (str): Path to save the processed image.
        resize_width (int, optional): New width (maintains aspect ratio if height is None).
        resize_height (int, optional): New height (maintains aspect ratio if width is None).
        quality (int, optional): Compression quality (1-100) for JPEG/WebP (default: 80).
    """
    try:
        # Open image
        img = Image.open(input_path)

        # Resize if needed
        if resize_width or resize_height:
            img = img.resize((resize_width, resize_height), Image.LANCZOS)

        # Save image with compression
        img.save(output_path, quality=quality, optimize=True)
        print(f"✅ Image saved to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"❌ Error processing image: {e}")

# B8: Audio Transcription (Dummy Implementation)
def B8(audio_file):
    # Load Whisper model (options: "tiny", "base", "small", "medium", "large")
    model = whisper.load_model("base")  # Change model size if needed

    # Transcribe audio
    #result = model.transcribe("your-audio-file.mp3")
    result = model.transcribe("/data/Sports.mp3")

    # Print the transcribed text
    print("Transcription:", result["text"])
# B9: Convert Markdown to HTML
def B9(md_file, html_file):
    print(f"Converting {md_file} to {html_file}")
    
    if not md_file or not md_file.endswith(".md") or not html_file or not html_file.endswith(".html"):
        raise HTTPException(status_code=400, detail=f"Input file ({md_file}) must be a Markdown file and output file ({html_file}) must be an HTML file")
    
    """Convert a Markdown file to an HTML file"""
    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_output = markdown.markdown(md_text)

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"Converted {md_file} to {html_file}")
    return html_file


# B10: API Endpoint for Filtering CSV and Returning JSON Data
def B10(CSV_FILE_PATH: str, targetfile: str, column: str, value: str):
    print(f"CSV_FILE_PATH: {CSV_FILE_PATH}, targetfile: {targetfile}, column: {column}, value: {value}")
    if not CSV_FILE_PATH or not CSV_FILE_PATH.endswith(".csv") or not column or not value:
        raise HTTPException(status_code=400, detail=f"Input file ({CSV_FILE_PATH}) must be a CSV file and column and value must be provided")
    
    """
    API endpoint to filter a CSV file based on a column and value.

    Query Parameters:
    - column: The column name to filter.
    - value: The value to match in the specified column.

    Returns:
    - JSON data with matching rows.
    """
    try:
        # Load CSV
        df = pd.read_csv(CSV_FILE_PATH)

        # Check if column exists
        if column not in df.columns:
            return {"error": f"Column '{column}' not found in CSV file"}

        # Filter data
        filtered_df = df[df[column].astype(str) == value]

        # Convert to JSON
        result = filtered_df.to_dict(orient="records")
        # save the result to a json file
        if targetfile:
            filtered_df.to_json(targetfile, orient="records", indent=4)
            return targetfile
        else:
            return json.loads(json.dumps(result))  # Convert NumPy types to JSON serializable format

    except Exception as e:
        return {"error": str(e)}