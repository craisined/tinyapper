from flask import (
    Flask,
    render_template,
    redirect,
    Response,
    request,
    stream_with_context,
    url_for,
)
from pathlib import Path
from parsetokens import wikitext_to_html
import sys
import time

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.parent))
from infer import load_model, stream_model

app = Flask(__name__)
HOST = "0.0.0.0"
PORT = 8000

model = load_model(current_dir / "static" / "yapperpedia.pt")


@app.route("/")
def index():
    return render_template("index.html")


# TODO: switch to streaming output
@app.route("/search", methods=["GET"])
def article():
    topic = request.args.get("query", default=None, type=str)
    return render_template("article.html", title=topic.lower(), query=topic.lower())


@app.route("/stream", methods=["POST"])
def stream_inference():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")

    def generate():
        rendered_text = prompt + "\n"
        total_text = ""
        for token in stream_model(prompt + "\n", model, max_tokens=900):
            total_text += token
            if token[-1] == "\n":
                rendered_text += total_text
                total_text = ""
            yield f"data: {(wikitext_to_html(rendered_text) + total_text).lower()}\n\n"
        rendered_text += total_text + "\n= max context! ="
        yield f"data: {(wikitext_to_html(rendered_text)).lower()}\n\n"

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
