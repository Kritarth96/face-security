import cv2
import time
import psutil
import threading
import numpy as np
from insightface.app import FaceAnalysis
from uniface.spoofing import MiniFASNet
from collections import deque

from recognition.recognizer import recognize_face

# ----------------------------------------
# Initialize SCRFD (buffalo_l)
# buffalo_sc is faster if you want more headroom on CPU
# ----------------------------------------
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------------------
# (Optional) Swap detector to mask-aware SCRFD
# Download scrfd_10g_gnkps_mask.onnx from:
# https://github.com/deepinsight/insightface/releases
# Then uncomment the two lines below:
# from insightface.model_zoo import get_model
# app.det_model = get_model("scrfd_10g_gnkps_mask", providers=["CPUExecutionProvider"])
# ----------------------------------------

# ----------------------------------------
# Initialize MiniFASNet
# ----------------------------------------
spoofer = MiniFASNet()

LIVE_THRESHOLD_NORMAL = 0.84   # full bare face
LIVE_THRESHOLD_MASKED = 0.60   # relaxed — lower-face texture is hidden
VOTE_WINDOW = 15
MIN_VOTES = 3
liveness_votes = deque(maxlen=VOTE_WINDOW)

# ----------------------------------------
# Open Webcam
# ----------------------------------------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ----------------------------------------
# Recognition State
# ----------------------------------------
recognition_status = None
recognition_distance = None
recognition_time = 0
status_message = ""
liveness_label = "WARMING UP"
liveness_color = (0, 165, 255)

# ----------------------------------------
# Background Recognition
# ----------------------------------------
recognition_running = False
recognition_lock = threading.Lock()

# ----------------------------------------
# SCRFD Optimization
# ----------------------------------------
frame_count = -1
last_faces = []
last_verification_time = 0
VERIFICATION_INTERVAL = 1.0

# ----------------------------------------
# Performance Metrics
# ----------------------------------------
last_frame_time = time.perf_counter()
current_fps = 0.0
process = psutil.Process()
total_cpu_usage = 0
cpu_samples = 0
total_ram_usage = 0
ram_samples = 0


# ----------------------------------------
# Mask Detection
# ----------------------------------------
def has_mask(face, frame):
    """
    Infers whether the detected face is wearing a mask using two signals:
    1. Landmark geometry — mouth landmarks appear suspiciously high when a mask is worn.
    2. Texture variance in the lower half of the face bbox — masks produce low-variance
       uniform regions vs. the natural skin/stubble texture of a bare face.

    InsightFace 5-point keypoints order: [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    kps = face.kps  # shape (5, 2)
    if kps is None:
        return False

    x1, y1, x2, y2 = face.bbox.astype(int)
    face_height = y2 - y1
    if face_height <= 0:
        return False

    # --- Signal 1: landmark geometry ---
    # On a bare face, mouth corners sit ~65–80% down the bbox.
    # With a mask, the model either fails to find them or pushes them upward.
    lmouth_y_rel = (kps[3][1] - y1) / face_height
    rmouth_y_rel = (kps[4][1] - y1) / face_height
    avg_mouth_y = (lmouth_y_rel + rmouth_y_rel) / 2

    if avg_mouth_y < 0.55:
        return True

    # --- Signal 2: texture variance in lower half ---
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(frame.shape[1], x2)
    y2c = min(frame.shape[0], y2)
    mid_y = y1c + (y2c - y1c) // 2
    lower_half = frame[mid_y:y2c, x1c:x2c]

    if lower_half.size == 0:
        return False

    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    variance = float(np.var(gray))

    # Low variance = uniform flat surface (mask fabric / surgical material)
    # Threshold tuned empirically; raise it if you get false positives on dark skin
    return variance < 180


# ----------------------------------------
# Eye-Region Crop for Masked Faces
# ----------------------------------------
def get_eye_region_crop(frame, bbox, kps):
    """
    When a mask is present, ArcFace embeddings degrade because the lower-face
    anchor points are occluded.  This crops only the forehead-to-nose band
    (top ~55% of the bbox) and pads it to a square so the recognizer
    doesn't receive an odd aspect ratio.
    """
    x1, y1, x2, y2 = bbox.astype(int)
    face_height = y2 - y1

    eye_y2 = y1 + int(face_height * 0.55)

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(frame.shape[1], x2)
    eye_y2c = min(frame.shape[0], eye_y2)

    crop = frame[y1c:eye_y2c, x1c:x2c]
    if crop.size == 0:
        return None

    # Pad to square (letterbox style, black fill)
    h, w = crop.shape[:2]
    side = max(h, w)
    padded = np.zeros((side, side, 3), dtype=np.uint8)
    padded[:h, :w] = crop
    return padded


# ----------------------------------------
# Background Verification Function
# ----------------------------------------
def recognize_face_thread(face_image, is_masked):
    global recognition_running, recognition_status
    global recognition_distance, recognition_time, status_message

    verify_start = time.perf_counter()
    try:
        result = recognize_face(face_image, "embeddings/embeddings.pkl")
        verify_elapsed = time.perf_counter() - verify_start

        with recognition_lock:
            recognition_status = result["recognized"]
            recognition_distance = result["distance"]
            recognition_time = verify_elapsed

            name = result["name"] if recognition_status else "UNKNOWN"
            mask_tag = " [MASK]" if is_masked else ""
            status_message = f"{name}{mask_tag}"

    except Exception as e:
        print("Verification Failed:", e)
        with recognition_lock:
            recognition_status = None
            recognition_distance = None
            recognition_time = 0
            status_message = "ERROR"
    finally:
        with recognition_lock:
            recognition_running = False


# ----------------------------------------
# Main Loop
# ----------------------------------------
while True:
    success, frame = camera.read()
    if not success:
        break

    # ----------------------------------------
    # Run SCRFD every 8th frame
    # ----------------------------------------
    frame_count += 1
    if frame_count % 8 == 0:
        last_faces = app.get(frame)

    # ----------------------------------------
    # Detect largest face & crop
    # ----------------------------------------
    face_crop = None
    largest_face = None
    largest_area = -1
    bbox_xyxy = None

    for face in last_faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_face = face
            bbox_xyxy = (x1, y1, x2, y2)

    # Track mask state for this frame
    masked = False

    if largest_face is not None:
        x1, y1, x2, y2 = bbox_xyxy
        face_crop = frame[y1:y2, x1:x2]

        # ----------------------------------------
        # Mask Detection
        # ----------------------------------------
        masked = has_mask(largest_face, frame)
        live_threshold = LIVE_THRESHOLD_MASKED if masked else LIVE_THRESHOLD_NORMAL
        mask_tag = " [MASK]" if masked else ""

        # ----------------------------------------
        # MiniFASNet Liveness Check (every frame)
        # Threshold is relaxed when a mask is detected so that the
        # missing lower-face texture doesn't cause false SPOOF verdicts.
        # ----------------------------------------
        spoof_result = spoofer.predict(frame, largest_face.bbox)
        is_confident_live = spoof_result.is_real and spoof_result.confidence > live_threshold
        liveness_votes.append(1 if is_confident_live else 0)

        if len(liveness_votes) < VOTE_WINDOW:
            liveness_verdict = None
            liveness_label = f"WARMING UP ({len(liveness_votes)}/{VOTE_WINDOW}){mask_tag}"
            liveness_color = (0, 165, 255)
        elif sum(liveness_votes) >= MIN_VOTES:
            liveness_verdict = True
            liveness_label = f"LIVE ({spoof_result.confidence:.2f}){mask_tag}"
            liveness_color = (0, 255, 0)
        else:
            liveness_verdict = False
            liveness_label = f"SPOOF ({spoof_result.confidence:.2f}){mask_tag}"
            liveness_color = (0, 0, 255)
            with recognition_lock:
                status_message = "SPOOF DETECTED"
                recognition_status = None

        # Draw face box with liveness color
        cv2.rectangle(frame, (x1, y1), (x2, y2), liveness_color, 2)

        # Draw a small MASK badge above the box when detected
        if masked:
            badge_x, badge_y = x1, max(0, y1 - 22)
            cv2.rectangle(frame, (badge_x, badge_y), (badge_x + 60, badge_y + 20), (255, 140, 0), -1)
            cv2.putText(frame, "MASK", (badge_x + 4, badge_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    else:
        liveness_votes.clear()
        liveness_verdict = None
        liveness_label = "NO FACE"
        liveness_color = (0, 255, 255)
        masked = False

    # ----------------------------------------
    # Recognition (only if liveness confirmed)
    # When masked, use the eye-region crop instead of the full face crop
    # so that ArcFace doesn't waste embedding capacity on mask pixels.
    # ----------------------------------------
    if time.time() - last_verification_time >= VERIFICATION_INTERVAL:
        last_verification_time = time.time()

        if face_crop is not None and liveness_verdict is True:
            with recognition_lock:
                should_start = not recognition_running
                if should_start:
                    recognition_running = True

            if should_start:
                if masked:
                    eye_crop = get_eye_region_crop(frame, largest_face.bbox, largest_face.kps)
                    face_copy = eye_crop.copy() if eye_crop is not None else face_crop.copy()
                else:
                    face_copy = face_crop.copy()

                threading.Thread(
                    target=recognize_face_thread,
                    args=(face_copy, masked),
                    daemon=True,
                ).start()

        elif face_crop is None:
            with recognition_lock:
                recognition_status = None
                recognition_distance = None
                status_message = "NO FACE"

    # ----------------------------------------
    # Draw Liveness Label
    # ----------------------------------------
    cv2.putText(frame, liveness_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, liveness_color, 2)

    # ----------------------------------------
    # Draw Recognition Result
    # ----------------------------------------
    with recognition_lock:
        current_distance = recognition_distance
        current_time = recognition_time
        current_message = status_message

    if current_message:
        if current_message in ("UNKNOWN", "SPOOF DETECTED"):
            color = (0, 0, 255)
        elif current_message in ("NO FACE", "WARMING UP"):
            color = (0, 255, 255)
        elif current_message == "ERROR":
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)

        cv2.putText(frame, current_message, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if current_distance is not None:
            # Show a warning if distance is higher than normal (expected with mask)
            dist_warning = " (mask penalty)" if masked and current_distance > 0.45 else ""
            cv2.putText(frame, f"Distance: {current_distance:.3f}{dist_warning}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ----------------------------------------
    # Performance Metrics
    # ----------------------------------------
    current_time_value = time.perf_counter()
    frame_delta = current_time_value - last_frame_time
    last_frame_time = current_time_value

    if frame_delta > 0:
        measured_fps = 1.0 / frame_delta
        current_fps = current_fps * 0.9 + measured_fps * 0.1

    current_cpu = psutil.cpu_percent(interval=None)
    total_cpu_usage += current_cpu
    cpu_samples += 1

    current_ram = process.memory_info().rss / (1024 * 1024)
    total_ram_usage += current_ram
    ram_samples += 1

    average_cpu = total_cpu_usage / cpu_samples
    average_ram = total_ram_usage / ram_samples

    cv2.putText(frame, f"FPS: {current_fps:.2f}",
                (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"CPU: {average_cpu:.1f}%",
                (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"RAM: {average_ram:.1f} MB",
                (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Verify: {current_time:.2f}s",
                (20, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()