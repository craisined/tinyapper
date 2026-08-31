from flask import Flask, render_template, redirect, request, url_for
from pathlib import Path
from parsetokens import wikitext_to_html
import sys

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.parent))
from infer import load_model, run_model

app = Flask(__name__)
HOST = "0.0.0.0"
PORT = 8000

model = load_model(current_dir / "static" / "yapperpedia.pt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["GET"])
def article():
    topic = request.args.get("query", default=None, type=str)
    if topic is None:
        return redirect(url_for("index"))
    article = run_model(f"= {topic} =\n", model, max_tokens=512)
    print(article)
    article = wikitext_to_html(article).lower()
    return render_template("article.html", title=topic.lower(), page_content=article)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
