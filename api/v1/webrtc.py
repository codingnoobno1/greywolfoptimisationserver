"""
Greywolf WebRTC Output API.

Architecture:
  INPUT  path: Flutter → binary WebSocket → push_frame() → AI    (NOT handled here)
  OUTPUT path: ResultStore → VideoProcessedTrack → WebRTC → Flutter RTCVideoView

Flutter sends a receive-only offer (no video tracks added from Flutter side).
Server responds with an answer that contains the AI-processed video track.

Critical fix:
  _pcs dict keeps RTCPeerConnection objects alive.
  Old code had `pc` as a local variable → Python GC destroyed it after the
  route returned → connection died within 20-30 seconds.
"""
import asyncio
import cv2
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from services.result_store import store
from core.logger import logger
from config.settings import settings

router = APIRouter()

# ── PC registry ───────────────────────────────────────────────────────────────
# Keeps PeerConnection objects alive (prevents Python GC from destroying them).
# Key: source_id (device UUID)  Value: RTCPeerConnection
_pcs: dict[str, RTCPeerConnection] = {}


# ── Output track ──────────────────────────────────────────────────────────────

class VideoProcessedTrack(VideoStreamTrack):
    """
    Pulls the AI-annotated frame for a device from ResultStore
    and sends it to the Flutter client via WebRTC at ~20 FPS.
    """
    kind = "video"

    def __init__(self, source_id: str):
        super().__init__()
        self.source_id = source_id
        self._pts = 0

    async def recv(self):
        # ~20 FPS output to save bandwidth
        await asyncio.sleep(1.0 / 20)

        img = store.get_latest_frame(self.source_id)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                img, f"WAITING: {self.source_id[:8]}...",
                (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2
            )

        from av import VideoFrame
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = self._pts
        frame.time_base = 1 / 1000
        self._pts += 50  # 50 ms steps → 20 FPS
        return frame


# ── Signaling endpoint ────────────────────────────────────────────────────────

@router.post("/offer/{source_id}")
async def webrtc_offer(source_id: str, request: Request):
    """
    Receive Flutter's SDP offer and return an answer.

    Flutter sends a receive-only offer (no tracks added from the phone).
    Server adds VideoProcessedTrack and replies with the answer SDP.
    The processed video then flows Server → Flutter.
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # ── Close any existing PC for this device ────────────────────────────────
    if source_id in _pcs:
        old_pc = _pcs.pop(source_id)
        try:
            await old_pc.close()
            logger.info(f"WebRTC: Closed stale PC for {source_id}")
        except Exception:
            pass

    # ── Create new PC and store it ───────────────────────────────────────────
    pc = RTCPeerConnection()
    _pcs[source_id] = pc          # <-- keeps it alive, prevents GC

    # ── Add the AI-processed output track ────────────────────────────────────
    pc.addTrack(VideoProcessedTrack(source_id))

    # ── Lifecycle logging ────────────────────────────────────────────────────
    @pc.on("iceconnectionstatechange")
    async def on_ice():
        logger.info(f"WebRTC ICE [{source_id}]: {pc.iceConnectionState}")

    @pc.on("connectionstatechange")
    async def on_state():
        state = pc.connectionState
        logger.info(f"WebRTC PC [{source_id}]: {state}")
        if state in ("failed", "closed"):
            _pcs.pop(source_id, None)
            try:
                await pc.close()
            except Exception:
                pass

    # ── No inbound track expected (Flutter uses binary WS for input) ─────────
    # If Flutter accidentally sends a track, just ignore it.
    @pc.on("track")
    def on_track(track):
        logger.debug(f"WebRTC: Unexpected inbound track from {source_id} — ignoring")

    # ── SDP exchange ─────────────────────────────────────────────────────────
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info(f"WebRTC: Answer sent to {source_id} ✅")

    return JSONResponse(content={
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


@router.get("/status")
async def webrtc_status():
    """List active WebRTC connections."""
    return {
        "active_connections": list(_pcs.keys()),
        "count": len(_pcs)
    }
