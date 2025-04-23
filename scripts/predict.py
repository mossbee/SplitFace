import os
import torch
from torchvision import transforms
from src.data_loading.dataset import TwinFaceDataset
from src.models.transfg import TransFG
from src.utils.visualization import visualize_predictions

def load_model(model_path):
    model = TransFG()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, dataloader):
    predictions = []
    with torch.no_grad():
        for images, _ in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

def main(image_folder, model_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = TwinFaceDataset(image_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model = load_model(model_path)
    predictions = predict(model, dataloader)

    visualize_predictions(predictions)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict twin face identities.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    
    args = parser.parse_args()
    main(args.image_folder, args.model_path)