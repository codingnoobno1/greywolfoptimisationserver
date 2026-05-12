import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from services.result_store import store
from core.logger import logger
from config.settings import settings

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"Client connected via WebSocket: {websocket.client}")
    
    target_frame_time = 1.0 / settings.FPS_LIMIT
    
    try:
        while True:
            start_time = asyncio.get_event_loop().time()
            
            # Send latest metadata (bounding boxes, landmarks, fps)
            result = store.get_result()
            await websocket.send_json(result)
            
            # Sleep to maintain target FPS
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, target_frame_time - elapsed)
            await asyncio.sleep(sleep_time)
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
