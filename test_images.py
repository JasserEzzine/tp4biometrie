import cv2
import os

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Testing face detection on dataset images:\n")

for person in os.listdir("dataset"):
    person_path = os.path.join("dataset", person)
    if not os.path.isdir(person_path):
        continue
    
    print(f"--- {person} ---")
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  {img_name}: ❌ Cannot read image")
            continue
        
        print(f"  {img_name}: Image size = {img.shape}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print(f"    ❌ No face detected!")
            # Try with more relaxed parameters
            faces = face_detector.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
            if len(faces) > 0:
                print(f"    ⚠️  Face found with relaxed parameters: {len(faces)} face(s)")
        else:
            print(f"    ✅ Face detected: {len(faces)} face(s)")
    print()
