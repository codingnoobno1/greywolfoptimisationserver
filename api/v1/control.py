from fastapi import APIRouter, HTTPException
from services.result_store import store
from services.webcam_service import webcam_service
from services.mqtt_client import mqtt_client
from pydantic import BaseModel

router = APIRouter()

class SourceRequest(BaseModel):
    source: str # webcam, mqtt, mobile, none

class ModeRequest(BaseModel):
    mode: str # yolo, mediapipe, all, none

@router.get("/status/{source_id}")
async def get_system_status(source_id: str):
    """Returns the current configuration for a specific source"""
    return {
        "source_id": source_id,
        "active_mode": store.get_active_mode(source_id),
        "server_status": "operational"
    }

@router.post("/output/mode/{source_id}")
async def set_output_mode(source_id: str, request: ModeRequest):
    """Dynamically switches the AI processing mode for a specific source"""
    mode = request.mode.lower()
    if mode not in ["yolo", "mediapipe", "all", "none"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use: yolo, mediapipe, all, none")
    
    store.set_active_mode(source_id, mode)
    return {"status": "ok", "source_id": source_id, "mode": mode}
