from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import json
import subprocess
import os

app = Flask(__name__)
app.secret_key = 'secret'
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config.json')
SERVICE_NAME = "video-capture.service"

def read_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def write_config(data):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(data, f, indent=4)

def service_status():
    result = subprocess.run(
        ["systemctl", "--user", "is-active", SERVICE_NAME],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode().strip()

def control_service(action):
    subprocess.run(["systemctl", "--user", action, SERVICE_NAME])

@app.route('/')
def index():
    config = read_config()
    return render_template('index.html', config=config, status=service_status())

@app.route('/update', methods=['POST'])
def update():
    settings = {
        "fps": int(request.form['fps']),
        "width": int(request.form['width']),
        "height": int(request.form['height']),
        "exposure_time": int(request.form['exposure_time']),
        "min_free_mb": int(request.form['min_free_mb'])
    }
    starts = request.form.getlist('start')
    ends = request.form.getlist('end')
    periods = [{"start": s, "end": e} for s, e in zip(starts, ends) if s and e]

    new_config = {
        "settings": settings,
        "periods": periods
    }

    write_config(new_config)
    flash("Configuration enregistrée. Redémarrer le service pour appliquer les modifications.")
    return redirect(url_for('index'))

@app.route('/service/<action>')
def service_action(action):
    if action in ['start', 'stop', 'restart']:
        control_service(action)
    return redirect(url_for('index'))

@app.route('/status')
def status():
    return jsonify(status=service_status())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
