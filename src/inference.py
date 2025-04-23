import os
import torch
from torchvision import transforms
from models.transfg import TransFG
from data_loading.dataset import TwinFaceDataset

def load_model(model_path):
    model = TransFG()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def infer(model, image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
    return output

def main(image_path, model_path):
    model = load_model(model_path)
    output = infer(model, image_path)
    print("Inference output:", output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run inference on a twin face image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    args = parser.parse_args()
    
    main(args.image_path, args.model_path)