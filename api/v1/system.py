from fastapi import APIRouter
from services.result_store import store
from services.frame_queue import get_queue_size
from config.settings import settings

router = APIRouter()

@router.get("/health")
async def system_health():
    """
    Standardized System Health & Telemetry.
    """
    result = store.get_result()
    return {
        "status": "operational",
        "active_source": store.get_input_source(),
        "active_mode": store.get_active_mode(),
        "performance": {
            "fps": round(result.get("fps", 0.0), 2),
            "queue_size": get_queue_size(),
            "max_queue": settings.FRAME_QUEUE_MAXSIZE
        },
        "config": {
            "fps_limit": settings.FPS_LIMIT,
            "resolution": "640x480 (Normalized)"
        }
    }
