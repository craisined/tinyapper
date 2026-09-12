from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_inference():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    kv_cache = data.get("cache", "")
