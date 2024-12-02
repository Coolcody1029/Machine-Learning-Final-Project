import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class_names = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']


num_classes = len(class_names)
model = models.resnet50(pretrained=False)  
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        class_name = class_names[preds[0]]
    
    return class_name


sample_image_path = "C:/Users/major/Downloads/colonn917.jpeg"  
if os.path.exists(sample_image_path):
    prediction = predict_image(sample_image_path)
    print(f"The image is predicted as: {prediction}")
else:
    print(f"Sample image not found at: {sample_image_path}")
