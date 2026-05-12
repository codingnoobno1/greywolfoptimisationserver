# Greywolf AI Platform: Multi-Stream 3-Tier Architecture

This document describes the high-performance, decoupled 3-tier architecture designed for multi-source AI ingestion and universal distribution.

## 3-Tier Architecture Overview

### Tier 1: Ingestion Layer (Input API)
**Purpose**: Efficiently receive data from diverse sources without waiting for processing.
- **Independence**: Sources (Mobile, ESP32, Webcam) just "push" data. They do not need to know about the consumers.
- **Protocol**: WebRTC (Video), WebSocket (Frames), MQTT/REST (Sensors).
- **Endpoint**: `/api/input/stream/{source_id}`

### Tier 2: Storage & Logic Layer (Processing Middle-Layer)
**Purpose**: Decouple producers from consumers.
- **Memory Store**: All incoming frames and telemetry are stored in an in-memory dictionary keyed by `source_id`.
- **Parallel processing**: AI Workers pull from the queue, process, and update the memory store with "annotated" versions of the data.
- **State Management**: Tracks which streams are active and their current AI modes (YOLO, MediaPipe).

### Tier 3: Distribution Layer (Output API)
**Purpose**: Distribute processed data to any client (Flutter, Web Dashboard, IoT displays).
- **Unified Output**: Clients pull from the same memory store using a specific `source_id`.
- **Stream Discovery**: Clients can list all active sources and dynamically switch.
- **Endpoints**: `/api/output/stream/{source_id}` (Video) and `/api/output/telemetry/{source_id}` (Data).

---

## Multi-Source API Reference

### 1. Ingestion (Input API)
| Endpoint | Method | Payload | Description |
|----------|--------|---------|-------------|
| `/api/input/webrtc/{source_id}` | POST | SDP Offer | High-speed video ingestion for a specific source. |
| `/api/input/frame/{source_id}` | WS | Binary/JPEG | WebSocket frame ingestion for low-power devices. |
| `/api/input/sensor/{source_id}` | POST | JSON | Ingest raw telemetry (Voltage, Current). |

### 2. Distribution (Output API)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/output/list` | GET | List all active sources and their current status. |
| `/api/output/video/{source_id}` | GET | MJPEG stream of the **processed** (annotated) feed for `source_id`. |
| `/api/output/data/{source_id}` | WS | Real-time JSON results (Gestures, Detections, Telemetry). |

### 3. Control API
| Endpoint | Method | Payload | Description |
|----------|--------|---------|-------------|
| `/api/control/mode/{source_id}` | POST | `{"mode": "mediapipe"}` | Switch AI analysis mode for a specific source. |
| `/api/control/command/{source_id}` | POST | `{"cmd": "switch1_on"}` | Send IoT control signals (Switches, Regulators). |

---

## Gesture Control Logic (Standardized)
- **Switch 1**: `ON` / `OFF`
- **Switch 2**: `ON` / `OFF`
- **Regulator**: Mode `1`, `2`, `3`, `4`, `5` (Controlled by indexed gestures or hand-height).

## Directory Evolution
```text
greywolfgesture/
├── api/v1/             # Input and Output specific routers
├── services/           # ResultStore (Multi-slot Memory Map)
├── workers/            # Source-aware AI Workers
└── main.py             # Entry point with dynamic routing
```
