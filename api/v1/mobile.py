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

@router.get("/status")
async def get_mobile_status():
    """Returns general server and AI status for the mobile app"""
    return {
        "status": "online",
        "active_mode": store.get_active_mode(),
        "fps_limit": settings.MOBILE_FPS,
        "features": ["yolo", "mediapipe", "gesture"]
    }

@router.get("/stream")
async def mobile_video_stream():
    """Dedicated MJPEG stream for mobile application"""
    return StreamingResponse(
        mobile_mjpeg_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

async def mobile_mjpeg_generator():
    """Generator for high-quality mobile streaming"""
    target_frame_time = 1.0 / settings.MOBILE_FPS
    
    while True:
        start_time = asyncio.get_event_loop().time()
        
        frame = store.get_latest_frame()
        if frame is None:
            # Placeholder for no signal
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'WAITING FOR MOBILE FEED...', (140, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        flag, encoded_image = cv2.imencode(
            '.jpg', 
            frame, 
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.MOBILE_JPEG_QUALITY]
        )
        
        if flag:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   bytearray(encoded_image) + b'\r\n')
            
        elapsed = asyncio.get_event_loop().time() - start_time
        sleep_time = max(0, target_frame_time - elapsed)
        await asyncio.sleep(sleep_time)

from core.models import FramePacket

@router.websocket("/ws")
async def mobile_websocket_ingest(websocket: WebSocket):
    """
    Standardized Mobile Ingestion.
    Protocol: Binary JPEG frames.
    Metadata: Assumed from connection or prepended (future).
    """
    await websocket.accept()
    client_id = f"mobile_{websocket.client.host}"
    logger.info(f"Mobile standardized connection: {client_id}")
    
    # Auto-engage mobile session source (let the dashboard control the AI mode)
    store.set_input_source("mobile")
    
    try:
        while True:
            # 1. Receive binary JPEG
            data = await websocket.receive_bytes()
            
            # 2. Decode
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # 3. Wrap in FramePacket (Standardization)
                packet = FramePacket(
                    frame=frame,
                    source_id=client_id,
                    source_type="mobile",
                    fps=settings.MOBILE_FPS,
                    quality=settings.MOBILE_JPEG_QUALITY
                )
                push_frame(packet)
                
                # 4. Standardized Distribution Feedback (Metadata only)
                result = store.get_result()
                await websocket.send_json({
                    "status": "active",
                    "source": client_id,
                    "detections": result.get("detections", []),
                    "gestures": result.get("gestures", []),
                    "fps": round(result.get("fps", 0.0), 2)
                })
    except WebSocketDisconnect:
        logger.info(f"Mobile disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Mobile WS Error: {e}")
