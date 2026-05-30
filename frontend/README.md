# Frontend

Standalone HTML/CSS/JS UI for the Face Attendance System.

This folder is self-contained and does NOT modify any backend code.
The frontend talks to the team's FastAPI backend over HTTP.

## How to run

1. Start the backend (in a separate terminal, from the repo root):
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```

2. Open `frontend/index.html` in a browser.
   - Either double-click it (opens via `file://`), or
   - Serve the folder for cleaner behaviour:
     ```bash
     cd frontend
     python -m http.server 8080
     ```
     then open http://localhost:8080

## What it shows

| Area | Source |
|------|--------|
| Live camera feed | `GET /video` (MJPEG) |
| Register a face | `POST /register {name}` |
| Registered users list | `GET /identities` |
| Delete a user | `DELETE /identities/{name}` |
| Info panel (name / emotion / liveness) | `GET /status` *(optional — graceful fallback if absent)* |

## Config

Change the backend URL in `config.js`:
```js
window.APP_CONFIG = {
    API_BASE: 'http://localhost:8000',
    ...
};
```
