from fastapi import APIRouter, HTTPException

from models.schemas import ListingRequest
from services.openrouter_service import optimize_listing

router = APIRouter()

@router.post("/api/listing/optimize")
def optimize(listing: ListingRequest):
    try:
        return optimize_listing(listing)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )