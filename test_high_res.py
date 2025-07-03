import cv2
import torch
import numpy as np
import time
from tqdm import tqdm
from ultralytics import YOLO

# Chargement du modèle sur GPU
#model = YOLO("model_cls.pt").to("cuda")

#model = YOLO("model_honly.pt").to("cuda")
model = YOLO("yolov8s.pt").to("cuda")
# Chargement de la vidéo
cap = cv2.VideoCapture("1080p.mp4")

# Informations vidéo
width = 1920
height = 1080
fps = cap.get(cv2.CAP_PROP_FPS)

# Création du fichier de sortie
out = cv2.VideoWriter("sliding_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Paramètres de sliding window
window_size = 640
xstride = 480
ystride = 440


def get_windows(img, size=640, xstride=480, ystride=480):
    H, W = img.shape[:2]
    windows, coords = [], []
    for y in range(0, H - size + 1, ystride):
        for x in range(0, W - size + 1, xstride):
            patch = img[y:y+size, x:x+size]
            windows.append(img[y:y+size, x:x+size])
            coords.append((x, y))
    return windows, coords

def draw_sliding_windows(img, size=640, xstride=480, ystride=480):
    count = 0
    H, W = img.shape[:2]
    for y in range(0, H - size + 1, ystride):
        for x in range(0, W - size + 1, xstride):
            cv2.rectangle(img, (x, y), (x + size, y + size), (0, 255, 0), 2)
            cv2.putText(img, f"{count}", (x + 5, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            count += 1
    #print(f"Nombre total de tuiles : {count}")
    return img


# FPS tracking
frame_count = 0
total_start = time.time()


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


for _ in tqdm(range(total_frames), desc="Traitement de la vidéo"):
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    frame = cv2.resize(frame, (1920, 1080))

    # Dessine les fenêtres sur l'image originale
    frame_with_windows = draw_sliding_windows(frame.copy(), window_size, xstride, ystride)

    patches, positions = get_windows(frame, window_size, xstride, ystride)
    batch = [torch.from_numpy(patch).permute(2, 0, 1).float() / 255 for patch in patches]
    batch = torch.stack(batch).to("cuda")

    with torch.no_grad():
        results = model(batch, verbose=False)

    for i, result in enumerate(results):
        annotated = result.plot()
        x, y = positions[i]
        frame_with_windows[y:y+window_size, x:x+window_size] = annotated

    out.write(frame_with_windows)

    end = time.time()
    frame_count += 1
    current_fps = 1 / (end - start)
    #print(f"Frame {frame_count} — FPS instantané : {current_fps:.2f}", end='\r')

# FPS moyen
total_end = time.time()
total_time = total_end - total_start
avg_fps = frame_count / total_time

print(f"\nVidéo enregistrée : sliding_output.mp4")
print(f"FPS moyen global : {avg_fps:.2f} sur {frame_count} frames")

# Nettoyage
cap.release()
out.release()

