from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class ListingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: Optional[str] = None
    description: Optional[str] = None
    bullets: Optional[List[str]] = None
    subjectKeywords: Optional[List[str]] = None
    genericKeywords: Optional[str] = None


class ListingResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: Optional[str] = None
    description: Optional[str] = None
    bullets: Optional[List[str]] = None
    subjectKeywords: Optional[List[str]] = None
    genericKeywords: Optional[List[str]] = None