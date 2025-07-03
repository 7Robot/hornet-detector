import cv2
from pypylon import pylon
import numpy as np
import os
import signal
import sys
from datetime import datetime
import shutil

running = True

def signal_handler(sig, frame):
    global running
    print("\nArrêt demandé, fermeture propre…")
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


if not disk_space_ok(output_dir):
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        f.write(f"{datetime.now()} - Pas assez d'espace disque, arrêt immédiat.\n")
    sys.exit(1)


camera.Open()
camera.ExposureAuto.SetValue('Off')
camera.ExposureTime.SetValue(20000)
camera.GainAuto.SetValue('Off')
camera.Gain.SetValue(0)
camera.PixelFormat.SetValue("BayerGB8")
camera.Width.Value = width 
camera.Height.Value = height
camera.AcquisitionFrameRate.SetValue(fps)


with open(os.path.join(output_dir, "log.txt"), "a") as f:
    f.write(f"{datetime.now()} - Démarrage capture: {output_filename}\n")

camera.StartGrabbing()

try:
    while camera.IsGrabbing() and running:
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
finally:
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        f.write(f"{datetime.now()} - Fin capture\n")
    camera.StopGrabbing()
    camera.Close()
    out.release()
    cv2.destroyAllWindows()
