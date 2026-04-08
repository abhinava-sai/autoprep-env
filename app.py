from fastapi import FastAPI
from subprocess import run

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/reset")
@app.post("/reset")
def run_env():
    result = run(["python", "inference.py"], capture_output=True, text=True)
    return {"output": result.stdout}