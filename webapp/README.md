# Leaf Scope — Web Frontend

A simple local web app for the plant disease model: upload a photo or use
your Mac's camera, and get live predictions in the browser. No terminal
commands needed once it's running.

Backend: FastAPI (serves your trained model as an API)
Frontend: a single HTML file (upload button, camera capture, results with
confidence bars) — just open it in your browser, no build step.

## 1. Install the extra dependency

From the project root:

```bash
pip3 install -r requirements.txt
```

(`fastapi`, `uvicorn`, and `python-multipart` were added to requirements.txt
for this.)

## 2. Start the backend

Point it at any trained checkpoint you already have in `outputs/`
(e.g. `compare_alexnet.pth`, or your own `best_alexnet.pth`):

```bash
cd webapp/backend
python3 app.py --checkpoint ../../outputs/compare_alexnet.pth
```

You should see:
```
Loading checkpoint from ../../outputs/compare_alexnet.pth onto device: mps
Loaded alexnet with 15 classes. Ready.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

`device: mps` confirms it's using your Mac's Apple Silicon GPU. Leave this
terminal window running — it's your local server.

## 3. Open the frontend

In Finder, double-click `webapp/frontend/index.html` (or right-click →
Open With → your browser). It opens as a local file and talks to the
backend at `http://localhost:8000`.

You should see a green dot and "Model server online" at the top — that
confirms the frontend found the backend.

## 4. Use it

- **Upload a photo**: pick an existing leaf image from your Mac
- **Use camera**: your browser will ask for camera permission — allow it,
  then click **Capture**
- Either way, click **Analyze** to get the top-3 predicted diseases with
  confidence bars

## Notes

- The backend must stay running in its terminal window the whole time
  you're using the frontend — closing that terminal stops the API.
- If the browser blocks the camera, check the site's camera permission
  (usually a camera icon in the address bar) and allow it.
- Model only knows Tomato, Potato, and Pepper diseases (15 classes) —
  other plants will still get a best-guess match from that set, not an
  "unknown" result.
- To switch which trained model serves predictions, just restart the
  backend with a different `--checkpoint` path (e.g. `compare_vgg16.pth`
  to compare how VGG16 performs on the same live photos).
