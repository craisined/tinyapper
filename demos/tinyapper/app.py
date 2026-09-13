from flask import Flask, render_template, request, url_for
from flask_socketio import SocketIO, emit
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.parent))
from infer import load_model, run_model

app = Flask(__name__)
socketio = SocketIO(app)

model = load_model(current_dir / "static" / "tinyapper.pt")

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('connect')
def connect():
    print(request.sid())

@socketio.event
def chat(data):
    request_id = request.sid()
    prompt = f"<|user|>\n{data.get('prompt')}<|assistant|>\n"
    output = run_model(prompt, model)
    emit("output", {"msg": output})

@socketio.on('disconnect')
def connect():
    print(request.sid())

if __name__=="__main__":
    socketio.run(app, host='0.0.0.0', port=8000)