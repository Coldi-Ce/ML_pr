import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import os
import joblib

# Загрузка модели
model_path = 'model/face_recognition_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модель ResNet18, настроенная для эмбеддингов
class FaceEmbeddingNet(nn.Module):
    def __init__(self):
        super(FaceEmbeddingNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)

model = FaceEmbeddingNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Трансформации для изображения
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Детектор лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Загрузка эмбеддингов
embeddings_path = 'model/face_embeddings.pkl'
if os.path.exists(embeddings_path):
    known_embeddings, known_emails = joblib.load(embeddings_path)
else:
    known_embeddings, known_emails = [], []

def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = image[y:y + h, x:x + w]
    return face

def get_embedding(face_img):
    face_tensor = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_tensor).cpu().numpy()[0]
    return embedding

def recognize_user_by_face(image_path, threshold=0.7):
    image = cv2.imread(image_path)
    face = extract_face(image)
    if face is None:
        return None

    embedding = get_embedding(face)
    if not known_embeddings:
        return None

    distances = [np.linalg.norm(embedding - emb) for emb in known_embeddings]
    min_distance = min(distances)
    if min_distance < threshold:
        return known_emails[distances.index(min_distance)]
    return None

def add_new_embedding(image_path, email):
    global known_embeddings, known_emails
    image = cv2.imread(image_path)
    face = extract_face(image)
    if face is None:
        return False

    embedding = get_embedding(face)
    known_embeddings.append(embedding)
    known_emails.append(email)
    joblib.dump((known_embeddings, known_emails), embeddings_path)
    return True
