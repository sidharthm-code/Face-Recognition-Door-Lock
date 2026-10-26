import cv2
import serial
import time

# Arduino connection
ARDUINO_PORT = "COM6"
BAUD_RATE = 9600

# Face recognition settings
CONFIDENCE_THRESHOLD = 40
name_list = ["", "USER", "USER"]

# Authorized user IDs
AUTHORIZED_IDS = {1, 2}

# Connect to Arduino
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("Arduino connected on", ARDUINO_PORT)

except serial.SerialException as e:
    print("Could not connect to Arduino:", e)
    print("Check the COM port and close Arduino Serial Monitor.")
    raise SystemExit

# Initialize face recognition
video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

imgBackground = cv2.imread("background.png")

if not video.isOpened():
    print("Could not open webcam.")
    arduino.close()
    raise SystemExit

if facedetect.empty() or imgBackground is None:
    print("Could not load the required image files.")
    video.release()
    arduino.close()
    raise SystemExit

last_unlock_time = 0
UNLOCK_COOLDOWN = 6

try:
    while True:
        ret, frame = video.read()

        if not ret:
            print("Could not read webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            person_id, confidence = recognizer.predict(
                gray[y:y + h, x:x + w]
            )
            print("ID:", person_id, "Confidence:", round(confidence, 2))

            authorized = (
                confidence < CONFIDENCE_THRESHOLD
                and person_id in AUTHORIZED_IDS
                and 0 < person_id < len(name_list)
            )

            if authorized:
                name = name_list[person_id]
                color = (0, 255, 0)

                if time.time() - last_unlock_time >= UNLOCK_COOLDOWN:
                    print("Authorized:", name)
                    print("Sending unlock command...")

                    arduino.write(b"1")
                    arduino.flush()

                    last_unlock_time = time.time()
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), color, 2
            )

            cv2.putText(
                frame, name, (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

        frame = cv2.resize(frame, (730, 485))
        imgBackground[150:635, 55:785] = frame

        cv2.imshow("Face Recognition Door Lock", imgBackground)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    video.release()
    cv2.destroyAllWindows()
    arduino.close()
    print("System closed.")