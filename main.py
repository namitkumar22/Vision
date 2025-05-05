from ultralytics import YOLO
import cv2

from SpeechToText import RecognizeSpeech
from TextToSpeech import speak
from FrameDescription import get_description


def YOLO_Recognition(frame):
    model = YOLO("YOLO_Models\yolo11l.pt")

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = results.names[cls_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def capture_video(query):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = YOLO_Recognition(frame)
        cv2.imshow("YOLO Object Detection", frame)
        description = get_description(frame, query)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        return description

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    while True:
        try:
            query = RecognizeSpeech()
        except Exception as e:
            msg = "Did not recognized your query, please try again"
            print(msg)
            speak(msg)
            continue
        description = capture_video(query)
        print(description)
        speak(description)
        cv2.destroyAllWindows()