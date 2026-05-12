from fastapi import APIRouter
from services.result_store import store

router = APIRouter()

@router.get("/status")
async def get_status():
    return {"active_mode": store.get_active_mode()}

@router.post("/toggle")
async def toggle_yolo(enabled: bool):
    if enabled:
        store.set_active_mode("yolo")
        return {"message": "YOLO activated", "active_mode": "yolo"}
    else:
        if store.get_active_mode() == "yolo":
            store.set_active_mode("none")
        return {"message": "YOLO deactivated", "active_mode": store.get_active_mode()}

@router.get("/detections")
async def get_detections():
    if store.get_active_mode() == "yolo":
        return {"detections": store.get_result().get("detections", [])}
    return {"detections": []}
