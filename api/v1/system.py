"""
Greywolf System API — Internal telemetry + device/session registry queries.

Endpoints:
  GET /api/system/health                          — server health
  GET /api/system/devices                         — all known devices (past + current)
  GET /api/system/devices/{device_id}             — single device detail
  GET /api/system/devices/{device_id}/sessions    — full session history
  GET /api/system/sessions/active                 — currently live sessions
"""
from fastapi import APIRouter, HTTPException
from services.result_store import store
from services.frame_queue import get_queue_size
from services.db_service import db
from config.settings import settings

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  Health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
async def system_health():
    """Server health + pipeline telemetry."""
    live_sources = store.list_sources()
    return {
        "status": "operational",
        "live_device_count": len(live_sources),
        "live_devices": live_sources,
        "performance": {
            "queue_size": get_queue_size(),
            "max_queue": settings.FRAME_QUEUE_MAXSIZE
        },
        "config": {
            "fps_limit": settings.FPS_LIMIT,
            "mobile_fps": settings.MOBILE_FPS,
            "resolution": "640x480"
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Device registry  (SQLite — persistent, survives restarts)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/devices")
async def list_all_devices():
    """
    All devices ever seen — past and present.
    Combines SQLite registry (persistent) with live ResultStore status.
    """
    devices = db.get_all_devices()
    live_ids = set(store.list_sources())

    # Annotate each row with real-time streaming status
    for d in devices:
        d["is_streaming"] = d["device_id"] in live_ids
        if d["is_streaming"]:
            d["active_mode"] = store.get_active_mode(d["device_id"])
        else:
            d["active_mode"] = None

    return {
        "total": len(devices),
        "online": sum(1 for d in devices if d["status"] == "online"),
        "devices": devices
    }


@router.get("/devices/{device_id}")
async def get_device_detail(device_id: str):
    """Detail for one device including latest session info."""
    device = db.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")

    sessions = db.get_device_sessions(device_id)
    live = device_id in store.list_sources()

    return {
        **device,
        "is_streaming": live,
        "active_mode": store.get_active_mode(device_id) if live else None,
        "session_count": len(sessions),
        "latest_session": sessions[0] if sessions else None
    }


@router.get("/devices/{device_id}/sessions")
async def get_device_sessions(device_id: str):
    """Full session history for a device — newest first."""
    device = db.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")

    sessions = db.get_device_sessions(device_id)
    return {
        "device_id": device_id,
        "total_sessions": len(sessions),
        "sessions": sessions
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Active sessions  (currently connected, no disconnected_at)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/sessions/active")
async def get_active_sessions():
    """
    Sessions that are currently open (WebSocket still connected).
    Cross-references SQLite sessions with live ResultStore.
    """
    sessions = db.get_active_sessions()
    live_ids = set(store.list_sources())

    for s in sessions:
        s["is_streaming"] = s["device_id"] in live_ids

    return {
        "count": len(sessions),
        "sessions": sessions
    }
