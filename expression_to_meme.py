import cv2
import os
import numpy as np
import threading
from deepface import DeepFace

# --- CONFIGURATION ---
MEME_FOLDER = "memes"
# The size of the side panel where the meme appears
MEME_DISPLAY_SIZE = (400, 480)

EMOTION_FILES = {
    "happy": "happy.png",
    "sad": "sad.png",
    "angry": "angry.png",
    "surprise": "surprise.png",
    "fear": "fear.png",
    "disgust": "disgust.png",
    "neutral": "neutral.png"
}

# Global variables for threading
current_emotion = "neutral"
is_analyzing = False


def load_memes():
    """Loads and pre-resizes images to fit the side panel."""
    meme_images = {}
    # Create a default black image for missing files
    default_img = np.zeros((MEME_DISPLAY_SIZE[1], MEME_DISPLAY_SIZE[0], 3), dtype=np.uint8)

    for emotion, filename in EMOTION_FILES.items():
        path = os.path.join(MEME_FOLDER, filename)
        if os.path.exists(path):
            img = cv2.imread(path)  # Read standard BGR (no alpha needed for side-by-side)
            if img is not None:
                # Resize to fit the side panel exactly
                img = cv2.resize(img, MEME_DISPLAY_SIZE)
                meme_images[emotion] = img
            else:
                meme_images[emotion] = default_img
        else:
            meme_images[emotion] = default_img
    return meme_images


def analyze_emotion_thread(frame_copy):
    """
    Runs DeepFace in a separate thread so the video doesn't freeze.
    """
    global current_emotion, is_analyzing
    try:
        # Run analysis (enforce_detection=False prevents errors if face is lost)
        result = DeepFace.analyze(frame_copy, actions=['emotion'],
                                  enforce_detection=False,
                                  detector_backend='opencv')

        if isinstance(result, list):
            current_emotion = result[0]['dominant_emotion']
        else:
            current_emotion = result['dominant_emotion']

    except Exception as e:
        pass
    finally:
        is_analyzing = False


def main():
    global is_analyzing, current_emotion

    # 1. Load Resources
    print("Loading resources...")
    memes = load_memes()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 2. Setup Camera
    cap = cv2.VideoCapture(0)

    # Set camera resolution to standard 640x480
    cap.set(3, 640)
    cap.set(4, 480)

    print("System Ready. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Flip frame horizontally for a "mirror" effect
        frame = cv2.flip(frame, 1)

        # --- THREADED EMOTION DETECTION ---
        # Only start a new analysis if the previous one is finished.
        # This keeps the UI buttery smooth.
        if not is_analyzing:
            is_analyzing = True
            # Create a thread to run DeepFace
            threading.Thread(target=analyze_emotion_thread, args=(frame.copy(),), daemon=True).start()

        # --- VISUALIZATION ---

        # 1. Draw Face Box (Fast, runs every frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Put text above the box
            cv2.putText(frame, current_emotion.upper(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 2. Prepare the Meme Panel
        if current_emotion in memes:
            meme_panel = memes[current_emotion]
        else:
            # Fallback black panel
            meme_panel = np.zeros((MEME_DISPLAY_SIZE[1], MEME_DISPLAY_SIZE[0], 3), dtype=np.uint8)

        # 3. Resize camera frame to match meme panel height if necessary
        # (Standard webcam is usually 480 height, same as our config)
        if frame.shape[0] != MEME_DISPLAY_SIZE[1]:
            frame = cv2.resize(frame,
                               (int(frame.shape[1] * (MEME_DISPLAY_SIZE[1] / frame.shape[0])), MEME_DISPLAY_SIZE[1]))

        # 4. Stitch Images Side-by-Side (The "Canvas")
        # numpy.hstack stacks arrays horizontally
        combined_window = np.hstack((frame, meme_panel))

        # Show the result
        cv2.imshow('Meme Matcher', combined_window)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()