# AI Content Detector (P1)

Paste a URL, get an evidence-backed verdict on the page's text and images.

## Quickstart

1. `cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cp .env.example .env`
2. `cd backend && .venv/bin/uvicorn app.main:create_app --factory --reload`
3. `cd frontend && npm install && npm run dev`
