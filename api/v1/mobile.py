"""
Greywolf Mobile Ingestion API — Tier 1

Device identity model:
  - device_id is a stable UUID generated once on the Flutter device
    and stored in SharedPreferences.
  - Never use IP addresses as identity (IP changes, NAT collapses
    multiple devices to one public IP).

Lifecycle on WebSocket connect:
  1. db.register_device(device_id)   — upsert into SQLite devices table
  2. db.open_session(device_id)      — create a session row, get session_id

Lifecycle on WebSocket disconnect:
  1. db.close_session(session_id)    — stamp disconnected_at
  2. db.mark_device_offline(device_id)
  3. store.remove_source(device_id)  — clean in-memory state
"""
import asyncio
import time
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, Response
from services.result_store import store
from services.frame_queue import push_frame
from services.db_service import db
from config.settings import settings
from core.logger import logger
from core.models import FramePacket

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  Status
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status/{device_id}")
async def get_mobile_status(device_id: str):
    """Returns server + AI status for a specific device."""
    device = db.get_device(device_id)
    return {
        "status": "online",
        "device_id": device_id,
        "registered": device is not None,
        "first_seen": device["first_seen"] if device else None,
        "last_seen": device["last_seen"] if device else None,
        "active_mode": store.get_active_mode(device_id),
        "fps_limit": settings.MOBILE_FPS,
        "features": ["yolo", "mediapipe", "gesture"]
    }


@router.get("/devices")
async def list_live_devices():
    """
    Returns devices that are currently streaming (in-memory ResultStore).
    Fast — no DB query needed for live status.
    """
    live_sources = store.list_sources()
    return {
        "count": len(live_sources),
        "devices": [
            {
                "device_id": src,
                "active_mode": store.get_active_mode(src),
            }
            for src in live_sources
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Ingestion WebSocket  (device_id replaces source_id — same URL shape)
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/{device_id}")
async def mobile_websocket_ingest(websocket: WebSocket, device_id: str):
    """
    Binary frame ingestion from Flutter.

    Flutter must supply a stable UUID as device_id — generated once and
    persisted in SharedPreferences. Never send the phone's IP here.

    Protocol:
      Client → Server : raw JPEG bytes
      Server → Client : JSON ACK / telemetry ticks
    """
    await websocket.accept()

    # ── Tier 2.5: Register device + open session ──────────────────────────
    db.register_device(device_id)
    session_id = db.open_session(device_id)
    logger.info(f"[WS] Connected: device={device_id}  session={session_id}")

    # Send immediate ACK so Flutter knows it's live
    await websocket.send_json({
        "status": "connected",
        "device_id": device_id,
        "session_id": session_id
    })

    try:
        while True:
            try:
                # ── Tier 1: Receive binary JPEG ───────────────────────────
                data = await websocket.receive_bytes()

                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    packet = FramePacket(
                        frame=frame,
                        source_id=device_id,
                        source_type="mobile_binary",
                        fps=settings.MOBILE_FPS,
                        quality=settings.MOBILE_JPEG_QUALITY
                    )
                    push_frame(packet)
                else:
                    logger.warning(f"[WS] Invalid frame from {device_id}")

            except WebSocketDisconnect:
                raise
            except Exception as loop_e:
                logger.error(f"[WS] Loop error ({device_id}): {loop_e}")
                continue  # Don't kill the socket for one bad frame

    except WebSocketDisconnect:
        logger.info(f"[WS] Disconnected: device={device_id}  session={session_id}")
    except Exception as e:
        logger.error(f"[WS] Fatal error ({device_id}): {e}")
    finally:
        # ── Tier 2.5: Close session + clean in-memory state ───────────────
        db.close_session(session_id)
        db.mark_device_offline(device_id)
        store.remove_source(device_id)
        logger.info(f"[WS] Cleanup done: device={device_id}")


# ─────────────────────────────────────────────────────────────────────────────
#  Telemetry WebSocket
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/telemetry/{device_id}")
async def mobile_telemetry_ws(websocket: WebSocket, device_id: str):
    """
    Throttled AI result feed back to Flutter.
    Sends at 5 FPS (200 ms) to avoid network congestion.
    """
    await websocket.accept()
    try:
        while True:
            result = store.get_result(device_id)
            await websocket.send_json({
                "status": "streaming",
                "device_id": device_id,
                "detections": result.get("detections", []),
                "gestures": result.get("gestures", []),
                "fps": round(result.get("fps", 0.0), 2),
                "ts": time.time()
            })
            await asyncio.sleep(0.2)  # throttled to 5 FPS
    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Snapshot + Stream
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/snapshot/{device_id}")
async def mobile_snapshot(device_id: str):
    """
    Single JPEG snapshot for a device.
    Flutter polls this at high frequency instead of parsing MJPEG.
    """
    frame = store.get_latest_frame(device_id)
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"WAITING: {device_id}", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    _, encoded = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return Response(content=bytes(encoded), media_type="image/jpeg")


@router.get("/stream/{device_id}")
async def mobile_video_stream(device_id: str):
    """MJPEG distribution mirror for a specific device."""
    from api.v1.web import web_mjpeg_generator
    return StreamingResponse(
        web_mjpeg_generator(device_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Binary Video Feed WebSocket  (Server → Flutter output stream)
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/feed/{device_id}")
async def mobile_video_feed(websocket: WebSocket, device_id: str):
    """
    Pushes AI-annotated JPEG frames back to Flutter over WebSocket.

    Why not WebRTC:
      WebRTC requires STUN/TURN for NAT traversal.
      Mobile carrier symmetric NAT blocks ICE → connection always fails.
      This WebSocket uses the exact same TLS connection path that already works
      for frame ingestion — zero additional NAT issues.

    Protocol:
      Server → Client : raw JPEG bytes (binary)
      Target: 15 FPS (67ms interval) — enough for gesture display, low bandwidth

    Flutter side:
      Receive bytes → Image.memory(bytes) → display in widget
    """
    await websocket.accept()
    logger.info(f"[FEED] Video feed connected: device={device_id}")

    frame_interval = 1.0 / settings.FPS_LIMIT  # respects global FPS cap
    empty_frame_cache: bytes | None = None      # cache the waiting frame

    try:
        while True:
            loop_start = asyncio.get_event_loop().time()

            img = store.get_latest_frame(device_id)

            if img is not None:
                _, encoded = cv2.imencode(
                    ".jpg", img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY]
                )
                await websocket.send_bytes(bytes(encoded))
            else:
                # Send a placeholder frame if AI hasn't produced output yet
                if empty_frame_cache is None:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder, f"AI WARMING UP...", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2
                    )
                    _, enc = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    empty_frame_cache = bytes(enc)
                await websocket.send_bytes(empty_frame_cache)

            elapsed = asyncio.get_event_loop().time() - loop_start
            await asyncio.sleep(max(0.0, frame_interval - elapsed))

    except WebSocketDisconnect:
        logger.info(f"[FEED] Video feed disconnected: device={device_id}")
    except Exception as e:
        logger.error(f"[FEED] Error for {device_id}: {e}")

