from fastapi import APIRouter
from services.result_store import store
from services.frame_queue import get_queue_size

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "ok",
        "active_mode": store.get_active_mode(),
        "queue_size": get_queue_size(),
        "current_fps": store.get_result().get("fps", 0.0)
    }
