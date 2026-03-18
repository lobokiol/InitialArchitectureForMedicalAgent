from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import chat, threads
from app.api.routers import users
from app.api.routers import auth
from app.middleware.auth import AuthMiddleware
from app.core.logging import logger  # ensure logging configured


def create_app() -> FastAPI:
    app = FastAPI(title="Medical RAG Assistant")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(AuthMiddleware)

    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(threads.router)
    app.include_router(users.router)
    return app


app = create_app()


@app.get("/healthz")
async def healthz():
    logger.info("health check ping")
    return {"status": "ok"}
