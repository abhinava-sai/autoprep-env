from fastapi import FastAPI
from subprocess import run
import uvicorn

app = FastAPI()


@app.get("/")
def home():
    return {"status": "running"}


@app.get("/reset")
@app.post("/reset")
def run_env():
    result = run(["python", "inference.py"], capture_output=True, text=True)
    return {"output": result.stdout}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()