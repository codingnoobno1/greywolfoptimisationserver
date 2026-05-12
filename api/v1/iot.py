import cv2
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from services.frame_queue import push_frame
from core.models import FramePacket
from core.logger import logger

router = APIRouter()

@router.post("/frame/{source_id}")
async def iot_frame_ingest(source_id: str, request: Request):
    """
    Standardized IoT Edge Ingestion (ESP32-CAM).
    Protocol: HTTP POST Binary JPEG.
    Endpoint: /api/iot/frame/{source_id}
    """
    try:
        # 1. Receive binary payload (Raw JPEG bytes from ESP32)
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty frame data")

        # 2. Decode JPEG
        nparr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # 3. Push to AI Queue with dedicated source identity
            packet = FramePacket(
                frame=frame,
                source_id=source_id,
                source_type="iot_edge",
                metadata={"client_ip": request.client.host}
            )
            push_frame(packet)
            return {"status": "ok", "source": source_id, "size": len(body)}
        
        raise HTTPException(status_code=400, detail="Invalid JPEG data")
        
    except Exception as e:
        logger.error(f"IoT Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
