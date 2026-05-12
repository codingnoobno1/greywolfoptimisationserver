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

async def _consume_ingest(track, source_id):
    """Asynchronously consume frames from an inbound track"""
    try:
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            packet = FramePacket(
                frame=img,
                source_id=source_id,
                source_type="mobile_webrtc",
                fps=settings.MOBILE_FPS,
                quality=settings.MOBILE_JPEG_QUALITY
            )
            push_frame(packet)
    except Exception as e:
        logger.error(f"WebRTC Ingest Error for {source_id}: {e}")

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
    def __init__(self, source_id: str):
        super().__init__()
        self.source_id = source_id
        self.counter = 0

    async def recv(self):
        # Limit framerate of processed stream to ~20 FPS to save bandwidth/CPU
        await asyncio.sleep(0.05) 
        
        # Get annotated frame from store for THIS specific source
        img = store.get_latest_frame(self.source_id)
        
        if img is None:
            # Send a black placeholder if no frame yet
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"WAITING FOR {self.source_id}...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Convert numpy BGR to aiortc VideoFrame
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = self.counter
        frame.time_base = 1 / 1000  # ms
        self.counter += 1
        
        return frame

@router.post("/offer/{source_id}")
async def webrtc_offer(source_id: str, request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    
    # 1. Pre-add the Outbound (AI Feedback) track so it's included in the Answer SDP
    pc.addTrack(VideoProcessedTrack(source_id))
    
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE Connection State [{source_id}]: {pc.iceConnectionState}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Peer Connection State [{source_id}]: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            logger.info(f"WebRTC: Inbound video track detected for {source_id}")
            # 2. Handle Ingest: Subscribe to incoming track and push to AI pipeline
            # We don't need to add this to PC, we just need to consume it
            asyncio.ensure_future(_consume_ingest(relay.subscribe(track), source_id))

    # Set remote description (The Offer from Flutter)
    await pc.setRemoteDescription(offer)

    # Create local description (The Answer for Flutter)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        content={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    )
