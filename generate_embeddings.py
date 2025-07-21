import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np

# Настройки
data_dir = "static/lfw"  # Папка с фото пользователей
output_path = "model/face_embeddings.pkl"
model_path = "model/face_recognition_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем модель
model = models.resnet18(pretrained=False)
model.fc = nn.Identity()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

embeddings = {}

for person in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person)
    if not os.path.isdir(person_dir):
        continue

    images = os.listdir(person_dir)
    if not images:
        continue

    # Берём только первое изображение
    image_path = os.path.join(person_dir, images[0])
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy().flatten()

    embeddings[person] = embedding

# Сохраняем словарь
with open(output_path, 'wb') as f:
    pickle.dump(embeddings, f)

print(f"Сохранено: {len(embeddings)} embeddings в {output_path}")
