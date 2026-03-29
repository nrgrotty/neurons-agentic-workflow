import logging
from fastapi import FastAPI
from neurons_agentic_workflow.creative_editor.controller import router
from pathlib import Path

# stdlib logging kept for uvicorn/fastapi infrastructure messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

app = FastAPI(title="Neurons Creative Editor")
app.include_router(router)
THIS_DIR = Path(__file__).parent

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True, reload_dirs=[str(THIS_DIR)])
