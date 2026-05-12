import asyncio
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.result_store import store
from config.settings import settings
from core.logger import logger

router = APIRouter()

class ModeRequest(BaseModel):
    mode: str # yolo, mediapipe, all, none

@router.get("/list")
async def list_active_sources():
    """List all active sources for the web dashboard"""
    return {"sources": store.list_sources()}

@router.get("/stream/{source_id}")
async def web_video_stream(source_id: str):
    """Processed MJPEG stream for the web dashboard"""
    return StreamingResponse(
        web_mjpeg_generator(source_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

async def web_mjpeg_generator(source_id: str):
    target_frame_time = 1.0 / settings.FPS_LIMIT
    while True:
        start_time = asyncio.get_event_loop().time()
        frame = store.get_latest_frame(source_id)
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f'WAITING FOR {source_id}', (140, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        flag, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY])
        if flag:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
            
        elapsed = asyncio.get_event_loop().time() - start_time
        await asyncio.sleep(max(0, target_frame_time - elapsed))

@router.websocket("/ws/{source_id}")
async def web_metadata_ws(websocket: WebSocket, source_id: str):
    """Metadata distribution for the web dashboard"""
    await websocket.accept()
    try:
        while True:
            result = store.get_result(source_id)
            await websocket.send_json({
                "detections": result.get("detections", []),
                "gestures": result.get("gestures", []),
                "telemetry": result.get("telemetry", {}),
                "fps": round(result.get("fps", 0.0), 2),
                "source": source_id
            })
            await asyncio.sleep(1.0 / settings.FPS_LIMIT)
    except WebSocketDisconnect:
        pass

@router.get("/status/{source_id}")
async def get_source_status(source_id: str):
    return {
        "source_id": source_id,
        "active_mode": store.get_active_mode(source_id),
        "server_status": "operational"
    }

@router.post("/control/mode/{source_id}")
async def set_source_mode(source_id: str, request: ModeRequest):
    mode = request.mode.lower()
    if mode not in ["yolo", "mediapipe", "all", "none"]:
        raise HTTPException(status_code=400, detail="Invalid mode")
    store.set_active_mode(source_id, mode)
    return {"status": "ok", "source_id": source_id, "mode": mode}
