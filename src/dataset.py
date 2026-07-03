from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet statistics, matching the pretrained backbone the features come from.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MVTecDataset(Dataset):
    """Single MVTec AD category. Training loads only defect-free images; the test
    split loads every image, labelled 0 (good) or 1 (defective)."""

    def __init__(self, root: str, category: str, train: bool = True, image_size: int = 224):
        self.root = Path(root) / category
        self.train = train
        self.transform = get_transforms(image_size)
        self.image_paths, self.labels = self._load_paths()

    def _load_paths(self):
        paths, labels = [], []
        if self.train:
            for p in sorted((self.root / "train" / "good").glob("*.png")):
                paths.append(p)
                labels.append(0)
        else:
            for defect_dir in sorted((self.root / "test").iterdir()):
                label = 0 if defect_dir.name == "good" else 1
                for p in sorted(defect_dir.glob("*.png")):
                    paths.append(p)
                    labels.append(label)
        return paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx], str(self.image_paths[idx])
