import asyncio
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
from services.result_store import store
from services.frame_queue import push_frame
from config.settings import settings
from core.logger import logger

router = APIRouter()

@router.get("/status/{source_id}")
async def get_mobile_status(source_id: str):
    """Returns general server and AI status for a specific mobile source"""
    return {
        "status": "online",
        "source_id": source_id,
        "active_mode": store.get_active_mode(source_id),
        "fps_limit": settings.MOBILE_FPS,
        "features": ["yolo", "mediapipe", "gesture"]
    }

from core.models import FramePacket

@router.websocket("/ws")
async def mobile_websocket_ingest(websocket: WebSocket):
    """
    Standardized Mobile Ingestion.
    Protocol: Binary JPEG frames.
    """
    await websocket.accept()
    client_id = f"mobile_{websocket.client.host}"
    logger.info(f"Mobile WS: Connection from {client_id}")
    
    # Send immediate ACK for synchronization
    await websocket.send_json({"status": "connected", "client_id": client_id})
    
    try:
        while True:
            # 1. Receive binary JPEG
            data = await websocket.receive_bytes()
            
            # 2. Decode
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # 3. Wrap in FramePacket
                packet = FramePacket(
                    frame=frame,
                    source_id=client_id,
                    source_type="mobile",
                    fps=settings.MOBILE_FPS,
                    quality=settings.MOBILE_JPEG_QUALITY
                )
                push_frame(packet)
                
                # 4. Feedback (Metadata) specific to this mobile source
                result = store.get_result(client_id)
                await websocket.send_json({
                    "status": "active",
                    "source": client_id,
                    "detections": result.get("detections", []),
                    "gestures": result.get("gestures", []),
                    "fps": round(result.get("fps", 0.0), 2)
                })
    except WebSocketDisconnect:
        logger.info(f"Mobile WS: Disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Mobile WS: Error for {client_id}: {e}")

