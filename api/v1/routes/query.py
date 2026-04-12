from fastapi import APIRouter, HTTPException
from api.v1.services.query_service import query_documents
from api.v1.schemas.query_schema import QueryRequest, QueryResponse

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
 
    try:
        q = query_documents(
            request.query,

        )
        print(q)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return q
