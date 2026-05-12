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
            
            # Wrap the track to ingest frames
            pc.addTrack(VideoIngestTrack(relay.subscribe(track), client_id))

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
