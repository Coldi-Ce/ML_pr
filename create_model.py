import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import json

data_dir = 'static/lfw'
model_path = 'model/face_recognition_model.pth'
label_map_path = 'model/label_map.json'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        self.transform = transform
        label_id = 0
        for person in sorted(os.listdir(root_dir)):
            person_path = os.path.join(root_dir, person)
            if not os.path.isdir(person_path):
                continue
            self.label_map[label_id] = person
            for img in os.listdir(person_path):
                img_path = os.path.join(person_path, img)
                self.image_paths.append(img_path)
                self.labels.append(label_id)
            label_id += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.resnet18(weights='IMAGENET1K_V1')
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

dataset = FaceDataset(data_dir, transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceRecognitionModel(num_classes=len(dataset.label_map)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_sum/len(loader):.4f}, Accuracy: {acc:.2f}%")

def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Тестовая точность: {100 * correct / total:.2f}%")

train(model, train_loader, epochs=10)
evaluate(model, test_loader)

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), model_path)
with open(label_map_path, 'w') as f:
    json.dump(dataset.label_map, f)
print("Модель и label_map сохранены.")
