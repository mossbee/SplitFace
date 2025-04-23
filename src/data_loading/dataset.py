from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class TwinFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load images and labels
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                twin_dir = os.path.join(label_dir + '_twin')
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                for img_name in os.listdir(twin_dir):
                    img_path = os.path.join(twin_dir, img_name)
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Example of how to use the dataset
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
# dataset = TwinFaceDataset(root_dir='path/to/train_data', transform=transform)