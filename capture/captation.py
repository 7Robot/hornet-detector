import cv2
from pypylon import pylon
import numpy as np
import os
import signal
import sys
from datetime import datetime
from time import sleep
import shutil

import json

with open("config.json", "r") as f:
    config = json.load(f)

settings = config.get("settings", {})
fps = settings.get("fps", 5)
width = settings.get("width", 2592)
height = settings.get("height", 1944)
exposure_time = settings.get("exposure_time", 20000)
min_free_mb = settings.get("min_free_mb", 500)

acquisition_periods = [
    (datetime.strptime(p["start"], "%Y-%m-%d %H:%M:%S"),
     datetime.strptime(p["end"], "%Y-%m-%d %H:%M:%S"))
    for p in config["periods"]
]

running = True

def signal_handler(sig, frame):
    global running
    with open(os.path.join(os.path.expanduser("~/videos"), "log.txt"), "a") as f:
        f.write(f"{datetime.now()} - Arrêt demandé, fermeture propre…\n")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def disk_space_ok(path, min_free_mb=500):
    total, used, free = shutil.disk_usage(path)
    return free >= min_free_mb * 1024 * 1024

tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
 
for device in devices:
    print(device.GetFriendlyName())
if not devices:
    print("no device found")

camera = pylon.InstantCamera()
camera.Attach(tl_factory.CreateFirstDevice())

# convertisseur d'image
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

width = 2592
height = 1944
fps = 10

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.expanduser("~/videos")
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, f"capture_{timestamp}.avi")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))


if not disk_space_ok(output_dir, min_free_mb=min_free_mb):
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        f.write(f"{datetime.now()} - Pas assez d'espace disque, arrêt immédiat.\n")
    sys.exit(1)

try:
    for i, (start_time, end_time) in enumerate(acquisition_periods):
        if not running:
            break

        print(f"\nPériode {i+1} — Attente jusqu'à {start_time}")
        while datetime.now() < start_time and running:
            sleep(1)
        if not running:
            break

        # === Ouvrir et configurer la caméra ===
        camera.Open()
        camera.ExposureAuto.SetValue('Off')
        camera.ExposureTime.SetValue(exposure_time)
        camera.GainAuto.SetValue('Off')
        camera.Gain.SetValue(0)
        camera.PixelFormat.SetValue("BayerGB8")
        camera.Width.Value = width
        camera.Height.Value = height
        camera.AcquisitionFrameRate.SetValue(fps)
        camera.StartGrabbing()

        # === Créer le fichier vidéo pour cette période ===
        timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = os.path.join(output_dir, f"capture_{timestamp}.avi")
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        with open(os.path.join(output_dir, "log.txt"), "a") as f:
            f.write(f"{datetime.now()} - Démarrage capture: {output_filename}\n")

        print(f"Capture de {start_time} à {end_time}")
        while datetime.now() < end_time and camera.IsGrabbing() and running:
            grab_result = camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = converter.Convert(grab_result)
                img = image.GetArray()
                if not disk_space_ok(output_dir):
                    with open(os.path.join(output_dir, "log.txt"), "a") as f:
                        f.write(f"{datetime.now()} - Pas assez d'espace disque\n")
                    running = False
                    break
                out.write(img)
            else:
                with open(os.path.join(output_dir, "log.txt"), "a") as f:
                    f.write(f"{datetime.now()} - Echec\n")
                    f.write(f"Code d'erreur: {grab_result.GetErrorCode()}\n")
                    f.write(f"Message d'erreur: {grab_result.GetErrorDescription()}\n")
            grab_result.Release()

        out.release()
        camera.StopGrabbing()
        camera.Close()

        with open(os.path.join(output_dir, "log.txt"), "a") as f:
            f.write(f"{datetime.now()} - Fin capture période {i+1}\n")

finally:
    camera.Close()
    cv2.destroyAllWindows()
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        f.write(f"{datetime.now()} - Capture terminée.\n")