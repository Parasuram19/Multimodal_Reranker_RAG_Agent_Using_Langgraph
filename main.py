from fastapi import FastAPI
from api.v1.routes.admin import router as admin_router
from api.v1.routes.query import router as query_router

app = FastAPI(title="Financial RAG API", version="1.0.0")

app.include_router(admin_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok"}