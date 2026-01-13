import json
import subprocess
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

CONFIG_PATH = Path(__file__).parent.parent / "config.json"

SERVICE_NAME = "video-capture.service"

app = Flask(__name__)


def read_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def write_config(data):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=4)


def service_status():
    result = subprocess.run(  # noqa: S603
        ["/usr/bin/systemctl", "--user", "is-active", SERVICE_NAME], capture_output=True
    )
    return result.stdout.decode().strip()


def control_service(action):
    if action in ["start", "stop", "restart"]:
        subprocess.run(["/usr/bin/systemctl", "--user", action, SERVICE_NAME])  # noqa: S603


@app.route("/")
def index():
    config = read_config()
    return render_template("index.html", config=config, status=service_status())


@app.route("/update", methods=["POST"])
def update():
    settings = {
        "fps": int(request.form["fps"]),
        "width": int(request.form["width"]),
        "height": int(request.form["height"]),
        "exposure_time": int(request.form["exposure_time"]),
        "min_free_mb": int(request.form["min_free_mb"]),
    }
    starts = request.form.getlist("start")
    ends = request.form.getlist("end")
    periods = [
        {"start": s, "end": e} for s, e in zip(starts, ends, strict=False) if s and e
    ]

    new_config = {"settings": settings, "periods": periods}

    write_config(new_config)
    flash(
        "Configuration enregistrée. "
        "Redémarrer le service pour appliquer les modifications."
    )
    return redirect(url_for("index"))


@app.route("/service/<action>")
def service_action(action):
    control_service(action)
    return redirect(url_for("index"))


@app.route("/status")
def status():
    return jsonify(status=service_status())


if __name__ == "__main__":
    app.run(host="localhost", port=5000)
