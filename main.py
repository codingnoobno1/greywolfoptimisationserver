from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

from core.logger import logger
from services.mqtt_client import mqtt_client
from services.webcam_service import webcam_service
from workers.ai_worker import start_workers
from config.settings import settings
from services.db_service import db
from services.frame_queue import push_frame
from fastapi import UploadFile, File

# Routers
from api.v1.health import router as health_router
from api.v1.greywolf import router as yolo_router
from api.v1.handgesture import router as mediapipe_router

# Streaming
from streaming.mjpeg import mjpeg_generator
from streaming.websocket import websocket_endpoint
from streaming.ingestion import browser_ingest_endpoint

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Starting Greywolf AI Platform...')
    start_workers(settings.NUM_WORKERS)
    
    if settings.USE_WEBCAM:
        webcam_service.start()
    else:
        mqtt_client.start()
    
    db.log_event('SYSTEM', 'Platform Started')
    
    yield
    
    logger.info('Shutting down...')
    mqtt_client.stop()
    webcam_service.stop()

app = FastAPI(title='Greywolf AI', lifespan=lifespan)

# Add CORS Middleware for Flutter/React compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

templates = Jinja2Templates(directory='templates')
os.makedirs('static', exist_ok=True)
app.mount('/static', StaticFiles(directory='static'), name='static')

app.include_router(health_router, prefix='/api/v1')
app.include_router(yolo_router, prefix='/api/v1/greywolf')
app.include_router(mediapipe_router, prefix='/api/v1/handgesture')

# --- PAGE ROUTES ---

@app.get('/', response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name='index.html', 
        context={'active_page': 'home'}
    )

@app.get('/yolo', response_class=HTMLResponse)
async def yolo_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name='yolo.html', 
        context={'active_page': 'yolo'}
    )

@app.get('/mediapipe', response_class=HTMLResponse)
async def mediapipe_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name='mediapipe.html', 
        context={'active_page': 'mediapipe'}
    )

@app.get('/settings', response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name='settings.html', 
        context={'active_page': 'settings'}
    )

# --- STREAMING ROUTES ---

@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(
        mjpeg_generator(), 
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.websocket('/ws')
async def ws_endpoint(websocket: WebSocket):
    await websocket_endpoint(websocket)

@app.websocket('/ingest')
async def ingest_endpoint(websocket: WebSocket):
    await browser_ingest_endpoint(websocket)

@app.post('/api/v1/ingest/frame')
async def ingest_frame(request: Request):
    # Receive raw binary JPEG from ESP32 or other source
    import cv2
    import numpy as np
    
    body = await request.body()
    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is not None:
        push_frame(frame)
        return {'status': 'ok'}
    return {'status': 'error', 'message': 'Invalid frame data'}

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
