import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Core & Services
from core.logger import logger
from config.settings import settings
from services.mqtt_client import mqtt_client
from services.webcam_service import webcam_service
from services.result_store import store
from services.db_service import db
from workers.ai_worker import start_workers

# Standardized Routers
from api.v1.mobile import router as mobile_router
from api.v1.iot import router as iot_router
from api.v1.output import router as output_router
from api.v1.system import router as system_router
from api.v1.control import router as control_router
from api.v1.webrtc import router as webrtc_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Initializing Greywolf AI Platform (Standardized)...')
    
    # Start AI Processing Pipeline
    start_workers(settings.NUM_WORKERS)
    
    # Initialize Default Input Source
    if settings.USE_WEBCAM:
        webcam_service.start()
    else:
        mqtt_client.start()
    
    db.log_event('SYSTEM', 'Platform Started - Standardized Architecture Active')
    
    yield
    
    logger.info('Shutting down...')
    mqtt_client.stop()
    webcam_service.stop()

app = FastAPI(
    title='Greywolf AI Standardized Platform', 
    version='2.0.0',
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Static Files & Templates
templates = Jinja2Templates(directory='templates')
os.makedirs('static', exist_ok=True)
app.mount('/static', StaticFiles(directory='static'), name='static')

# --- STANDARDIZED ROUTE REGISTRATION ---

# Ingestion Layer (Input)
app.include_router(mobile_router, prefix='/api/mobile', tags=['Ingestion'])
app.include_router(iot_router, prefix='/api/iot', tags=['Ingestion'])

# Distribution Layer (Output)
app.include_router(output_router, prefix='/api/output', tags=['Distribution'])

# Management & Health
app.include_router(system_router, prefix='/api/system', tags=['System'])
app.include_router(control_router, prefix='/api', tags=['Control'])
app.include_router(webrtc_router, prefix='/api/webrtc', tags=['Ingestion'])

# --- UI PAGE ROUTES ---

@app.get('/', response_class=HTMLResponse, tags=['UI'])
async def dashboard_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name='dashboard.html', 
        context={'active_page': 'dashboard'}
    )

@app.get('/yolo', response_class=HTMLResponse, tags=['UI'])
async def yolo_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name='dashboard.html', 
        context={'active_page': 'vision', 'initial_mode': 'yolo'}
    )

@app.get('/mediapipe', response_class=HTMLResponse, tags=['UI'])
async def mediapipe_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name='dashboard.html', 
        context={'active_page': 'gestures', 'initial_mode': 'mediapipe'}
    )

@app.get('/settings', response_class=HTMLResponse, tags=['UI'])
async def settings_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name='settings.html', 
        context={'active_page': 'settings'}
    )

if __name__ == '__main__':
    uvicorn.run(
        'main:app', 
        host='0.0.0.0', 
        port=8443, 
        reload=True,
        ssl_keyfile='key.pem',
        ssl_certfile='cert.pem'
    )
