import os 
from PIL import Image
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms

class LucchiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        self.images = sorted([root_dir + '/img/' + i for i in os.listdir(os.path.join(root_dir, 'img'))])
        if os.path.exists(os.path.join(root_dir, 'img/.DS_Store')):
            self.images.remove(os.path.join(root_dir, 'img/.DS_Store'))

        self.masks = sorted([root_dir + '/label/' + i for i in os.listdir(os.path.join(root_dir, 'label'))])
        if os.path.exists(os.path.join(root_dir, 'label/.DS_Store')):
            self.masks.remove(os.path.join(root_dir, 'label/.DS_Store'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.transform(image)
        except OSError:
            print(f"error - {self.images[idx]}")
            return
        mask = Image.open(self.masks[idx]).convert('L')

        mask = self.transform(mask)

        return image, mask