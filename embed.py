import torch
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from PIL import Image
from torchvision import models

# Загрузка модели
def load_model(model_path):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(model.fc.in_features, 50)  # Укажите нужное количество классов
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Загрузка label_map (словаря с метками)
def load_label_map(label_map_path):
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    return label_map

# Извлечение эмбеддингов для всех изображений в датасете
def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    paths = []
    with torch.no_grad():
        for images, labels, image_paths in data_loader:
            images = images.to(device)
            output = model(images)
            embeddings.append(output.cpu().numpy())  # Сохраняем эмбеддинги
            paths.extend(image_paths)

    embeddings = np.vstack(embeddings)
    return embeddings, paths

# Сравнение эмбеддинга нового изображения с эмбеддингами из датасета
def recognize_user_by_face(model, image_path, label_map, embeddings, image_paths, device):
    # Преобразования для входного изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Загрузка и преобразование изображения
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Извлечение эмбеддинга
    model.eval()
    with torch.no_grad():
        emb = model(image).cpu().numpy().flatten()

    # Сравнение с существующими эмбеддингами
    similarities = cosine_similarity([emb], embeddings)
    closest_idx = similarities.argmax()
    confidence = similarities[0, closest_idx]

    # Если уверенность больше 0.6, возвращаем имя пользователя
    if confidence > 0.6:
        user_id = label_map.get(str(closest_idx), None)
        return user_id, confidence
    return None, None
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceFolderDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), 0, img_path
