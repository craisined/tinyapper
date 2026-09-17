from flask import Flask, render_template, request, url_for
from flask_socketio import SocketIO, emit
from pathlib import Path
import torch
import sys

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.parent))
from infer import load_model, run_model

app = Flask(__name__)
socketio = SocketIO(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, model_config = load_model(current_dir / "static" / "tinyapper.pt")
caches = {}
cache_dtype = torch.bfloat16 if device == "cuda" else torch.float32

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('connect')
def connect():
    request_id = request.sid
    caches[request_id] = model.create_kv_caches(batches=1, device=device, dtype=cache_dtype)

@socketio.event
def chat(data):
    request_id = request.sid
    cache = caches[request_id]
    prompt = f"<|user|>\n{data.get('prompt')}<|assistant|>\n"
    output = run_model(prompt, (model, model_config), cache=cache, max_tokens=1024-cache[0].total_tokens)
    emit("output", {"msg": output})
    if cache[0].total_tokens > 768:
        emit("output", {"msg": "cache more than 3/4 full, creating new cache"})
        caches[request_id] = model.create_kv_caches(batches=1, device=device, dtype=cache_dtype)

@socketio.on('disconnect')
def connect():
    caches.pop(request.sid, None)

if __name__=="__main__":
    socketio.run(app, host='0.0.0.0', port=8000)
