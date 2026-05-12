import asyncio
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from services.result_store import store
from config.settings import settings
from core.logger import logger

router = APIRouter()

@router.get("/list")
async def list_active_sources():
    """List all active sources in the memory store"""
    return {"sources": store.list_sources()}

@router.get("/stream/{source_id}")
async def standardized_video_stream(source_id: str):
    """
    Standardized Output Stream (MJPEG) for a specific source.
    """
    return StreamingResponse(
        output_mjpeg_generator(source_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

async def output_mjpeg_generator(source_id: str):
    """Generator for high-quality rendered MJPEG output"""
    target_frame_time = 1.0 / settings.FPS_LIMIT
    
    while True:
        start_time = asyncio.get_event_loop().time()
        
        # Get latest annotated frame from the specific source slot
        frame = store.get_latest_frame(source_id)
        if frame is None:
            # Placeholder for no signal
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f'WAITING FOR {source_id}', (140, 240), 
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

@router.websocket("/ws/{source_id}")
async def standardized_metadata_ws(websocket: WebSocket, source_id: str):
    """
    Standardized Metadata Distribution for a specific source.
    """
    await websocket.accept()
    logger.info(f"Metadata client connected to {source_id}: {websocket.client}")
    
    try:
        while True:
            # Send latest metadata for this source at target FPS
            result = store.get_result(source_id)
            data = {
                "detections": result.get("detections", []),
                "gestures": result.get("gestures", []),
                "telemetry": result.get("telemetry", {}),
                "fps": round(result.get("fps", 0.0), 2),
                "source": source_id
            }
            await websocket.send_json(data)
            await asyncio.sleep(1.0 / settings.FPS_LIMIT)
            
    except WebSocketDisconnect:
        logger.info(f"Metadata client disconnected from {source_id}")
    except Exception as e:
        logger.error(f"Output WS Error for {source_id}: {e}")
