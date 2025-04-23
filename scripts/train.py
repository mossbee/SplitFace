import os
import yaml
import torch
from torch.utils.data import DataLoader
from src.data_loading.dataset import TwinFaceDataset
from src.models.transfg import TransFG
from src.trainer import Trainer

def main():
    # Load configuration
    with open('configs/training.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset
    train_dataset = TwinFaceDataset(config['data']['train_dir'], transform=config['data']['transform'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Initialize model
    model = TransFG(num_classes=config['model']['num_classes']).to(device)

    # Initialize trainer
    trainer = Trainer(model, train_loader, config)

    # Start training
    trainer.train()

if __name__ == '__main__':
    main()