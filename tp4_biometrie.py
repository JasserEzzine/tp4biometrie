import cv2
import numpy as np
import os


class FaceRecognitionPCA:

    def __init__(self, n_components=30):
        self.n_components = n_components
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.mean = None
        self.eigenvectors = None
        self.projections = None
        self.labels = None

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        if len(faces) == 0:
            return None, None

        largest = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = largest

        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))

        return face, (x, y, w, h)

    def load_dataset(self, dataset_path):
        X = []
        y = []

        for person in os.listdir(dataset_path):

            person_path = os.path.join(dataset_path, person)

            if not os.path.isdir(person_path):
                continue

            for img_name in os.listdir(person_path):

                img_path = os.path.join(person_path, img_name)

                img = cv2.imread(img_path)

                if img is None:
                    continue

                face, _ = self.detect_face(img)

                if face is not None:
                    X.append(face.flatten())
                    y.append(person)

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        self.labels = y

        return X, y

    def compute_pca(self, X):

        self.mean = np.mean(X, axis=0)

        X_centered = X - self.mean

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        k = min(self.n_components, Vt.shape[0])

        self.eigenvectors = Vt[:k].T

        self.projections = np.dot(X_centered, self.eigenvectors)

    def project(self, face_vector):

        face_centered = face_vector - self.mean

        projection = np.dot(face_centered, self.eigenvectors)

        return projection

    def recognize(self, image_path, threshold=4000):

        img = cv2.imread(image_path)

        face, rect = self.detect_face(img)

        if face is None:
            return None, None, "No Face", img, None

        vector = face.flatten()

        proj = self.project(vector)

        distances = np.linalg.norm(self.projections - proj, axis=1)

        min_dist = np.min(distances)

        idx = np.argmin(distances)

        identity = self.labels[idx]

        decision = "Match" if min_dist < threshold else "No Match"

        return identity, min_dist, decision, img, rect


if __name__ == "__main__":

    dataset = "dataset"

    test_image = "test.jpg"

    model = FaceRecognitionPCA(n_components=30)

    X, y = model.load_dataset(dataset)

    model.compute_pca(X)

    identity, distance, decision, img, rect = model.recognize(test_image)

    print("Distance minimale :", distance)

    print("Identité prédite :", identity)

    print("Décision :", decision)

    if rect is not None:

        x, y, w, h = rect

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text1 = f"ID: {identity}"

    text2 = f"Distance: {distance:.2f}" if distance is not None else "Distance: N/A"

    text3 = f"Result: {decision}"

    cv2.putText(img, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(img, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(img, text3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("TP4 Biometrie - Resultat", img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
