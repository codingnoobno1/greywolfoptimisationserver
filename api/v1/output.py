import asyncio
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from services.result_store import store
from config.settings import settings
from core.logger import logger

router = APIRouter()

@router.get("/stream")
async def standardized_video_stream():
    """
    Standardized Output Stream (MJPEG).
    Renders AI results (YOLO + MediaPipe) onto the feed.
    """
    return StreamingResponse(
        output_mjpeg_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

async def output_mjpeg_generator():
    """Generator for high-quality rendered MJPEG output"""
    target_frame_time = 1.0 / settings.FPS_LIMIT
    
    while True:
        start_time = asyncio.get_event_loop().time()
        
        # Get latest annotated frame from the store
        frame = store.get_latest_frame()
        if frame is None:
            # Placeholder for no signal
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'NO ACTIVE SOURCE / STANDBY', (140, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        flag, encoded_image = cv2.imencode(
            '.jpg', 
            frame, 
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY]
        )
        
        if flag:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   bytearray(encoded_image) + b'\r\n')
            
        elapsed = asyncio.get_event_loop().time() - start_time
        sleep_time = max(0, target_frame_time - elapsed)
        await asyncio.sleep(sleep_time)

@router.websocket("/ws")
async def standardized_metadata_ws(websocket: WebSocket):
    """
    Standardized Metadata Distribution.
    Protocol: JSON only.
    Schema: { detections: [], gestures: [], fps: float, source: str }
    """
    await websocket.accept()
    logger.info(f"Metadata client connected: {websocket.client}")
    
    try:
        while True:
            # Send latest metadata at target FPS
            result = store.get_result()
            # result already contains detections, gestures, fps from store
            # we also add the current active source if available
            data = {
                "detections": result.get("detections", []),
                "gestures": result.get("gestures", []),
                "fps": round(result.get("fps", 0.0), 2),
                "source": store.get_input_source()
            }
            await websocket.send_json(data)
            await asyncio.sleep(1.0 / settings.FPS_LIMIT)
            
    except WebSocketDisconnect:
        logger.info(f"Metadata client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"Output WS Error: {e}")
