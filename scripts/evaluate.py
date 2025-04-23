import os
import torch
import yaml
from src.data_loading.dataset import TwinFaceDataset
from src.models.transfg import TransFG
from src.utils.metrics import calculate_accuracy
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def evaluate_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy

def main():
    config = load_config('configs/default.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TwinFaceDataset(config['data']['train_data'], transform=None)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)

    model = TransFG()
    model.load_state_dict(torch.load(config['model']['checkpoint_path']))
    model.to(device)

    accuracy = evaluate_model(model, dataloader, device)
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()