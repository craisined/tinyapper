from flask import Flask, render_template

app = Flask(__name__)
HOST = "0.0.0.0"
PORT = 8000

@app.route("/")
def index():
    return render_template("index.html")

if __name__=="__main__":
    app.run(host=HOST, port=PORT)