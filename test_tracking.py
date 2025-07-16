import cv2
import numpy as np
from pynput.mouse import Controller

# Configuration
TRACK_ID = 1
BBOX_WIDTH = 60
BBOX_HEIGHT = 30
ZONE_IGNORE = (100, 640, 1080, 80)  # x, y, w, h — zone non surveillée
MIN_BBOX_AREA = 1000
MAX_HISTORY = 5
STATIONARY_THRESHOLD_PX = 10

# Temps d'attente entre les frames
twait = 5  # ms
width, height = 1280, 720


mouse_pos = [width // 2, height // 2]  # position initiale


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos[0] = x
        mouse_pos[1] = y


cv2.namedWindow("Simulation frelon asiatique")
cv2.setMouseCallback("Simulation frelon asiatique", mouse_callback)


mouse = Controller()
track_history = []


def compute_frelon_score(memory):
    # Longueur cumulée des vols stationnaires
    total_stationary_time = sum(memory["stationary_durations"])
    average_bbox_size = np.mean(memory["bbox_sizes"]) if memory["bbox_sizes"] else 0
    nb_stationary_phases = len(memory["stationary_durations"])

    score = 0

    # Pondérations ajustables
    if total_stationary_time > 30:
        score += 0.4
    if nb_stationary_phases >= 2:
        score += 0.2
    if average_bbox_size > 1400:
        score += 0.3
    if memory["age"] > 40:
        score += 0.1

    return min(score, 1.0)


def is_stationary(history, threshold=10):
    if len(history) < 2:
        return False
    positions = np.array(history)
    dx = positions[:, 0].max() - positions[:, 0].min()
    dy = positions[:, 1].max() - positions[:, 1].min()
    return dx < threshold and dy < threshold


def is_in_ignore_zone(cx, cy, zone):
    zx, zy, zw, zh = zone
    return zx <= cx <= zx + zw and zy <= cy <= zy + zh


def analyze_movement(track):
    cx, cy = track["center"]
    area = track["bbox_area"]
    age = track["age"]

    if is_in_ignore_zone(cx, cy, ZONE_IGNORE):
        print(f"[IGNORÉ] Track {track['track_id']} dans zone non surveillée")
        return

    if area < MIN_BBOX_AREA:
        print(f"[IGNORÉ] Track {track['track_id']} trop petit (area={area})")
        return

    if is_stationary(track_history):
        print(f"[ALERTE] Track {track['track_id']} est STATIONNAIRE (age={age})")
    else:
        print(f"Track {track['track_id']} en mouvement")


def draw_overlay(frame, bbox, color=(0, 255, 0), label="Track 1"):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_ignore_zone(frame, zone):
    x, y, w, h = zone
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(
        frame,
        "Zone non surveillee",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )


# Fenêtre d'affichage
canvas = np.zeros((height, width, 3), dtype=np.uint8)

print("🖱️ Déplace la souris pour simuler un frelon asiatique.")
print("Appuie sur ESC pour quitter.")

age = 0

while True:
    frame = canvas.copy()
    draw_ignore_zone(frame, ZONE_IGNORE)

    # Obtenir position souris
    # mx, my = mouse.position
    mx, my = mouse_pos  # Utiliser la position de la souris capturée

    cx, cy = int(mx), int(my)

    print(f"Position souris: ({cx}, {cy})")

    # Simuler bbox autour de la souris
    x = cx - BBOX_WIDTH // 2
    y = cy - BBOX_HEIGHT // 2
    bbox = (x, y, BBOX_WIDTH, BBOX_HEIGHT)
    area = BBOX_WIDTH * BBOX_HEIGHT

    # Suivi et historique
    track_history.append((cx, cy))
    if len(track_history) > MAX_HISTORY:
        track_history.pop(0)

    track = {
        "track_id": TRACK_ID,
        "bbox": bbox,
        "center": (cx, cy),
        "bbox_area": area,
        "age": age,
    }

    draw_overlay(frame, bbox)
    analyze_movement(track)

    cv2.imshow("Simulation frelon asiatique", frame)
    key = cv2.waitKey(twait)
    age += 1

    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
