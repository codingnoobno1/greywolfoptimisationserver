import asyncio
import time
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

@router.websocket("/ws/{source_id}")
async def mobile_websocket_ingest(websocket: WebSocket, source_id: str):
    """
    Standardized Mobile Ingestion with Dynamic Source ID.
    """
    await websocket.accept()
    logger.info(f"Ingestion WS: Source '{source_id}' connected")
    
    # Send immediate ACK
    await websocket.send_json({"status": "connected", "source_id": source_id})
    
    try:
        while True:
            try:
                # 1. Receive binary JPEG (Throttle point)
                data = await websocket.receive_bytes()
                
                # 2. Decode
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # 3. Wrap in FramePacket & Push to Queue
                    packet = FramePacket(
                        frame=frame,
                        source_id=source_id,
                        source_type="mobile_binary",
                        fps=settings.MOBILE_FPS,
                        quality=settings.MOBILE_JPEG_QUALITY
                    )
                    push_frame(packet)
                else:
                    logger.warning(f"Mobile WS: Received empty/invalid frame from {source_id}")
                    
            except WebSocketDisconnect:
                raise # Re-raise to outer catch
            except Exception as loop_e:
                logger.error(f"Mobile Ingestion Loop Error: {loop_e}")
                continue # Don't kill the socket for one bad frame
    except WebSocketDisconnect:
        logger.info(f"Ingestion WS: Source '{source_id}' disconnected")
    except Exception as e:
        logger.error(f"Ingestion WS: Error for {source_id}: {e}")

@router.websocket("/ws/telemetry/{source_id}")
async def mobile_telemetry_ws(websocket: WebSocket, source_id: str):
    """
    Standardized Throttled Telemetry for Mobile.
    Sends AI results at 5 FPS (200ms) to avoid network congestion.
    """
    await websocket.accept()
    try:
        while True:
            result = store.get_result(source_id)
            await websocket.send_json({
                "status": "streaming",
                "detections": result.get("detections", []),
                "gestures": result.get("gestures", []),
                "fps": round(result.get("fps", 0.0), 2),
                "ts": time.time()
            })
            await asyncio.sleep(0.2) # Throttled to 5 FPS
    except WebSocketDisconnect:
        pass

@router.get("/stream/{source_id}")
async def mobile_video_stream(source_id: str):
    """
    Dedicated Mobile Distribution Mirror.
    Provides a high-priority stream path specifically for Flutter devices.
    """
    from api.v1.web import web_mjpeg_generator
    return StreamingResponse(
        web_mjpeg_generator(source_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

