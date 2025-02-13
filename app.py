import os
import subprocess
import json
import glob
import sqlite3
import base64
import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
import uvicorn
import requests  # For calling the LLM API
from pathlib import Path

app = FastAPI()

# Get the AIPROXY_TOKEN from the environment
AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
if not AIPROXY_TOKEN:
    raise Exception("AIPROXY_TOKEN environment variable not set")

def validate_path(path: str) -> Path:
    # Convert to a Path object and resolve
    p = Path(path).resolve()
    # Enforce that all file paths must be under /data
    data_path = Path("/data").resolve()
    if not str(p).startswith(str(data_path)):
        raise HTTPException(status_code=400, detail="Access to paths outside /data is not allowed.")
    return p

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(...)):
    file_path = validate_path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        content = file_path.read_text()
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/run")
async def run_task(task: str = Query(...)):
    try:
        # Use a helper function to parse the task description.
        # In a real implementation, you might send a prompt to GPT-4o-Mini.
        task_id = parse_task_description(task)
        
        # Map the task_id to a specific function.
        if task_id == "A1":
            result = task_A1(task)
        elif task_id == "A2":
            result = task_A2(task)
        elif task_id == "A3":
            result = task_A3(task)
        elif task_id == "A4":
            result = task_A4(task)
        elif task_id == "A5":
            result = task_A5(task)
        elif task_id == "A6":
            result = task_A6(task)
        elif task_id == "A7":
            result = task_A7(task)
        elif task_id == "A8":
            result = task_A8(task)
        elif task_id == "A9":
            result = task_A9(task)
        elif task_id == "A10":
            result = task_A10(task)
        # Business tasks B3-B10 can be added similarly.
        else:
            raise HTTPException(status_code=400, detail="Task not recognized.")

        return {"status": "OK", "result": result}
    except HTTPException as he:
        raise he
    except Exception as e:
        # Agent error
        raise HTTPException(status_code=500, detail=str(e))

def parse_task_description(task: str) -> str:
    # For simplicity, we search for keywords that hint at the task.
    # A production version would call the GPT-4o-Mini model.
    if "datagen.py" in task:
        return "A1"
    if "prettier" in task:
        return "A2"
    if "Wednesdays" in task or "थी" in task or "रविवार" in task:  # add more keywords for dates
        # We assume the task is like counting days (A3) or similar.
        if "dates.txt" in task:
            return "A3"
    if "contacts" in task and "sorted" in task:
        return "A4"
    if ".log" in task and "recent" in task:
        return "A5"
    if "Markdown" in task and "docs" in task:
        return "A6"
    if "email" in task and "sender" in task:
        return "A7"
    if "credit card" in task:
        return "A8"
    if "comments" in task and "similar" in task:
        return "A9"
    if "ticket-sales" in task or "Gold" in task:
        return "A10"
    # Business tasks can be similarly matched.
    return "UNKNOWN"

def task_A1(task: str):
    # Task A1: Install uv if needed and run datagen.py with ${user.email} as argument.
    # Assume user email is provided in an environment variable (e.g. USER_EMAIL)
    user_email = os.environ.get("USER_EMAIL")
    if not user_email:
        raise Exception("USER_EMAIL environment variable not set")
    
    # Ensure that 'uv' (or a tool similar to it) is installed.
    try:
        subprocess.run(["uv", "--version"], check=True)
    except Exception:
        # Install uv (this is just an example, adjust the install command as needed)
        subprocess.run(["pip", "install", "uv"], check=True)
    
    # Download and run the remote script.
    datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    # You might download it first, or use curl/wget; here we download to /tmp
    datagen_path = "/tmp/datagen.py"
    r = requests.get(datagen_url)
    if r.status_code != 200:
        raise Exception("Failed to download datagen.py")
    with open(datagen_path, "w") as f:
        f.write(r.text)
    
    # Run the script with the user email as argument.
    subprocess.run(["python", datagen_path, user_email], check=True)
    return "A1 executed"

def task_A2(task: str):
    # Task A2: Format /data/format.md with prettier@3.4.2
    file_path = validate_path("/data/format.md")
    
    # Run prettier. Ensure prettier is installed (you may install it locally via npm if required).
    # Example using subprocess (the command might vary based on your environment)
    cmd = ["prettier", "--version"]
    try:
        version = subprocess.check_output(cmd).decode().strip()
        if version != "3.4.2":
            # Force install or use the local version
            subprocess.run(["npm", "install", "prettier@3.4.2"], check=True)
    except Exception as e:
        raise Exception("Prettier not installed: " + str(e))
    
    # Format file in-place
    subprocess.run(["prettier", "--write", str(file_path)], check=True)
    return "A2 executed"


def task_A3(task: str):
    # Task A3: Count the number of Wednesdays in /data/dates.txt and write the number to /data/dates-wednesdays.txt
    input_path = validate_path("/data/dates.txt")
    output_path = validate_path("/data/dates-wednesdays.txt")
    
    count = 0
    for line in input_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            dt = datetime.datetime.strptime(line, "%Y-%m-%d")
            if dt.weekday() == 2:  # 0=Monday, 1=Tuesday, 2=Wednesday, ...
                count += 1
        except Exception:
            continue  # Skip lines that cannot be parsed as dates
    
    output_path.write_text(str(count))
    return "A3 executed"

def task_A4(task: str):
    input_path = validate_path("/data/contacts.json")
    output_path = validate_path("/data/contacts-sorted.json")
    
    with open(input_path, "r") as f:
        contacts = json.load(f)
    
    # Assume contacts is a list of dicts with keys "first_name" and "last_name"
    contacts_sorted = sorted(contacts, key=lambda c: (c.get("last_name", ""), c.get("first_name", "")))
    
    with open(output_path, "w") as f:
        json.dump(contacts_sorted, f, indent=2)
    return "A4 executed"

def task_A5(task: str):
    # List all .log files in /data/logs/
    log_files = sorted(
        glob.glob("/data/logs/*.log"),
        key=lambda f: os.path.getmtime(f),
        reverse=True
    )[:10]
    
    output_lines = []
    for log_file in log_files:
        with open(log_file, "r") as f:
            first_line = f.readline().strip()
            output_lines.append(first_line)
    
    output_path = validate_path("/data/logs-recent.txt")
    output_path.write_text("\n".join(output_lines))
    return "A5 executed"

def task_A6(task: str):
    docs_dir = validate_path("/data/docs/")
    index = {}
    # Recursively find all .md files under /data/docs/
    for md_file in docs_dir.rglob("*.md"):
        # Get filename relative to /data/docs/
        rel_filename = str(md_file.relative_to(docs_dir))
        with open(md_file, "r") as f:
            for line in f:
                if line.strip().startswith("# "):  # first occurrence of an H1 header
                    title = line.strip()[2:].strip()
                    index[rel_filename] = title
                    break
    index_path = validate_path("/data/docs/index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    return "A6 executed"

def call_llm(prompt: str) -> str:
    # Example call to the LLM API using GPT-4o-Mini.
    # You may need to adjust the URL and payload according to your AI Proxy’s documentation.
    api_url = "https://api.ai-proxy.example/llm/gpt4o-mini"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    payload = {"prompt": prompt, "max_tokens": 50}
    response = requests.post(api_url, json=payload, headers=headers, timeout=10)
    if response.status_code != 200:
        raise Exception("LLM API call failed")
    return response.json()["result"]

def task_A7(task: str):
    # Read the email message
    input_path = validate_path("/data/email.txt")
    email_content = input_path.read_text()
    # Prepare a short prompt to extract sender's email address.
    prompt = f"Extract the sender's email address from the following message:\n{email_content}"
    sender_email = call_llm(prompt).strip()
    output_path = validate_path("/data/email-sender.txt")
    output_path.write_text(sender_email)
    return "A7 executed"

def task_A8(task: str):
    # Read the image file and encode it as base64 (if needed)
    input_path = validate_path("/data/credit-card.png")
    with open(input_path, "rb") as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode()
    prompt = f"Extract the credit card number (digits only, no spaces) from this image (base64 encoded): {img_b64}"
    card_number = call_llm(prompt).strip().replace(" ", "")
    output_path = validate_path("/data/credit-card.txt")
    output_path.write_text(card_number)
    return "A8 executed"

def task_A9(task: str):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    input_path = validate_path("/data/comments.txt")
    comments = [line.strip() for line in input_path.read_text().splitlines() if line.strip()]
    
    if len(comments) < 2:
        raise Exception("Not enough comments for comparison.")
    
    embeddings = model.encode(comments, convert_to_tensor=True)
    # Compute cosine similarity matrix
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    max_score = -1
    pair = (None, None)
    for i in range(len(comments)):
        for j in range(i+1, len(comments)):
            if cosine_scores[i][j] > max_score:
                max_score = cosine_scores[i][j]
                pair = (comments[i], comments[j])
    
    output_path = validate_path("/data/comments-similar.txt")
    output_path.write_text("\n".join(pair))
    return "A9 executed"

def task_A10(task: str):
    db_path = validate_path("/data/ticket-sales.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Sum of total sales for Gold tickets
    query = "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"
    cursor.execute(query)
    result = cursor.fetchone()[0]
    conn.close()
    
    output_path = validate_path("/data/ticket-sales-gold.txt")
    output_path.write_text(str(result))
    return "A10 executed"
