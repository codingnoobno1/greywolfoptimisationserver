from fastapi import APIRouter
from services.result_store import store

router = APIRouter()

@router.get("/status")
async def get_status():
    return {"active_mode": store.get_active_mode()}

@router.post("/toggle")
async def toggle_mediapipe(enabled: bool):
    if enabled:
        store.set_active_mode("mediapipe")
        return {"message": "MediaPipe activated", "active_mode": "mediapipe"}
    else:
        if store.get_active_mode() == "mediapipe":
            store.set_active_mode("none")
        return {"message": "MediaPipe deactivated", "active_mode": store.get_active_mode()}

@router.get("/gestures")
async def get_gestures():
    if store.get_active_mode() == "mediapipe":
        return {"gestures": store.get_result().get("gestures", [])}
    return {"gestures": []}
