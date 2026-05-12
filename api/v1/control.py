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

@router.get("/status")
async def get_system_status():
    """Returns the current input and output configuration"""
    return {
        "input_source": store.get_input_source(),
        "active_mode": store.get_active_mode(),
        "server_status": "operational"
    }

@router.post("/input/source")
async def set_input_source(request: SourceRequest):
    """Dynamically switches the video input source"""
    source = request.source.lower()
    if source not in ["webcam", "mqtt", "mobile", "none"]:
        raise HTTPException(status_code=400, detail="Invalid source. Use: webcam, mqtt, mobile, none")
    
    # 1. Stop all current hardware ingestion
    webcam_service.stop()
    mqtt_client.stop()
    
    # 2. Start the requested source
    if source == "webcam":
        webcam_service.start()
    elif source == "mqtt":
        mqtt_client.start()
    elif source == "mobile":
        # Mobile is reactive (WebSocket based), no active start needed here
        pass
    
    store.set_input_source(source)
    return {"status": "ok", "source": source}

@router.post("/output/mode")
async def set_output_mode(request: ModeRequest):
    """Dynamically switches the AI processing mode"""
    mode = request.mode.lower()
    if mode not in ["yolo", "mediapipe", "all", "none"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use: yolo, mediapipe, all, none")
    
    store.set_active_mode(mode)
    return {"status": "ok", "mode": mode}
