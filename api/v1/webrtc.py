import asyncio
import json
import os
import cv2
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay

from services.frame_queue import push_frame
from services.result_store import store
from core.models import FramePacket
from core.logger import logger
from config.settings import settings

router = APIRouter()
relay = MediaRelay()

from av import VideoFrame

class VideoIngestTrack(VideoStreamTrack):
    """
    A video stream track that receives frames from WebRTC 
    and pushes them into the AI processing pipeline.
    """
    def __init__(self, track, client_id):
        super().__init__()
        self.track = track
        self.client_id = client_id

    async def recv(self):
        frame = await self.track.recv()
        
        # Convert aiortc frame to numpy/OpenCV format
        img = frame.to_ndarray(format="bgr24")
        
        # Standardize and Push to AI Pipeline
        packet = FramePacket(
            frame=img,
            source_id=self.client_id,
            source_type="mobile_webrtc",
            fps=settings.MOBILE_FPS,
            quality=settings.MOBILE_JPEG_QUALITY
        )
        push_frame(packet)
        
        return frame

class VideoProcessedTrack(VideoStreamTrack):
    """
    A video stream track that pulls the latest processed frame
    from the ResultStore and sends it back to the client.
    """
    def __init__(self):
        super().__init__()
        self.counter = 0

    async def recv(self):
        # Limit framerate of processed stream to ~20 FPS to save bandwidth/CPU
        await asyncio.sleep(0.05) 
        
        # Get annotated frame from store
        img = store.get_latest_frame()
        
        if img is None:
            # Send a black placeholder if no frame yet
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "WAITING FOR AI...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert numpy BGR to aiortc VideoFrame
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = self.counter
        frame.time_base = 1 / 1000  # ms
        self.counter += 1
        
        return frame

@router.post("/offer")
async def webrtc_offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    client_id = f"webrtc_{request.client.host}"
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"WebRTC connection state is {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()

    @pc.on("track")
    def on_track(track):
      if track.kind == "video":
        logger.info(f"WebRTC: Received video track from {client_id}")
        # Set input source to mobile for the dashboard
        store.set_input_source("mobile")
        
        # 1. Ingest incoming stream
        pc.addTrack(VideoIngestTrack(relay.subscribe(track), client_id))
        
        # 2. Return processed stream (with MediaPipe drawings)
        pc.addTrack(VideoProcessedTrack())

    # Set remote description
    await pc.setRemoteDescription(offer)

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        content={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    )
