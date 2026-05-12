import cv2
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from services.frame_queue import push_frame
from core.models import FramePacket
from core.logger import logger

router = APIRouter()

@router.post("/frame")
async def iot_frame_ingest(request: Request):
    """
    Standardized IoT Ingestion (ESP32-CAM).
    Protocol: HTTP POST Binary JPEG.
    Resolution: Optimized for ESP32 (usually 320x240).
    """
    try:
        # 1. Receive binary payload
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty frame data")

        # 2. Decode JPEG
        nparr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # 3. Wrap in FramePacket
            client_ip = request.client.host
            packet = FramePacket(
                frame=frame,
                source_id=f"iot_{client_ip}",
                source_type="iot",
                metadata={"protocol": "http_post"}
            )
            push_frame(packet)
            return {"status": "ok", "received": len(body)}
        
        raise HTTPException(status_code=400, detail="Invalid JPEG data")
        
    except Exception as e:
        logger.error(f"IoT Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
