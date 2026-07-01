from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.listing_routes import router

app = FastAPI(
    title="Amazon Listing Optimizer"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router)


@app.get("/")
def health_check():
    return {
        "status": "running"
    }