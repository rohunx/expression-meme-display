import cv2
import os
import numpy as np
import threading
from deepface import DeepFace
from collections import deque, Counter

# ------- SETTINGS -------
MEME_FOLDER = "memes"
EMOJI_FOLDER = "emojis"
MEME_DISPLAY_SIZE = (400, 480)

SHOW_BOUNDING_BOX = True     # Toggle bounding box ON/OFF
SHOW_EMOJIS = True            # <-- NEW: Toggle emoji overlay ON/OFF

EMOTION_FILES = {
    "happy": "happy.png",
    "angry": "angry.png",
    "neutral": "neutral.png"
}

ALLOWED_EMOTIONS = ["happy", "angry", "neutral"]

# Globals
is_analyzing = False
emotion_buffer = deque(maxlen=6)
current_display_emotion = "neutral"


def load_memes():
    meme_images = {}
    for emotion, filename in EMOTION_FILES.items():
        path = os.path.join(MEME_FOLDER, filename)
        img = cv2.imread(path)
        if img is not None:
            meme_images[emotion] = cv2.resize(img, MEME_DISPLAY_SIZE)
        else:
            meme_images[emotion] = np.zeros((MEME_DISPLAY_SIZE[1], MEME_DISPLAY_SIZE[0], 3))
    return meme_images


def load_emojis():
    emojis = {}
    for emotion, filename in EMOTION_FILES.items():
        path = os.path.join(EMOJI_FOLDER, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # RGBA loading
        emojis[emotion] = img
    return emojis


def overlay_emoji(frame, emoji_img, x, y, w, h):
    if emoji_img is None:
        return frame

    emoji = cv2.resize(emoji_img, (w, h))

    emoji_x = x
    emoji_y = y - int(h * 0.15)
    emoji_y = max(0, emoji_y)

    if emoji.shape[2] == 4:
        rgba = emoji[:, :, :3]
        alpha = emoji[:, :, 3] / 255.0
    else:
        rgba = emoji
        alpha = np.ones((emoji.shape[0], emoji.shape[1]))

    for c in range(3):
        frame[emoji_y:emoji_y+h, emoji_x:emoji_x+w, c] = (
            frame[emoji_y:emoji_y+h, emoji_x:emoji_x+w, c] * (1 - alpha)
            + rgba[:, :, c] * alpha
        )

    return frame


def analyze_emotion_thread(frame_copy):
    global is_analyzing, current_display_emotion

    try:
        res = DeepFace.analyze(
            frame_copy,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )

        if isinstance(res, list):
            res = res[0]

        scores = res['emotion']
        filtered = {k: v for k, v in scores.items() if k in ALLOWED_EMOTIONS}
        dom = max(filtered, key=filtered.get)

        emotion_buffer.append(dom)
        current_display_emotion = Counter(emotion_buffer).most_common(1)[0][0]

    except:
        pass
    finally:
        is_analyzing = False


def main():
    global is_analyzing

    memes = load_memes()
    emojis = load_emojis()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    print("System running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if not is_analyzing:
            is_analyzing = True
            threading.Thread(target=analyze_emotion_thread, args=(frame.copy(),), daemon=True).start()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:

            must_show_box = SHOW_BOUNDING_BOX or current_display_emotion == "neutral"

            if must_show_box:
                color = (255, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, current_display_emotion.upper(),
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # ===== EMOJI DRAWING BASED ON BOOLEAN =====
            if SHOW_EMOJIS:
                emoji_img = emojis.get(current_display_emotion)
                frame = overlay_emoji(frame, emoji_img, x, y, w, h)
            # If SHOW_EMOJIS=False → skip overlay entirely

        # ---- Meme region ----
        meme_panel = memes.get(current_display_emotion, memes["neutral"])

        if frame.shape[0] != MEME_DISPLAY_SIZE[1]:
            scale = MEME_DISPLAY_SIZE[1] / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), MEME_DISPLAY_SIZE[1]))

        combined = np.hstack((frame, meme_panel))
        cv2.imshow("Meme Matcher", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
