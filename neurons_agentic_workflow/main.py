from fastapi import FastAPI
from neurons_agentic_workflow.creative_editor.controller import router
from pathlib import Path

app = FastAPI(title="Neurons Creative Editor")
app.include_router(router)
THIS_DIR = Path(__file__).parent

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True, reload_dirs=[str(THIS_DIR)])
