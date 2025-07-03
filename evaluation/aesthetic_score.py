from pathlib import Path
import clip
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import tqdm

class MLP(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        pil_image = Image.open(img_path)
        image = self.preprocess(pil_image)
        return image, str(img_path)  # Convert Path to string

def custom_collate_fn(batch):
    images, paths = zip(*batch)
    return torch.stack(images, 0), paths

def aes_score(folder_path, model_path="resources/sac+logos+ava1-l14-linearMSE.pth", batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(768).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    model2, preprocess = clip.load("ViT-L-14", device=device)
    path = Path(folder_path)
    images = list(path.rglob("*.png"))

    if not images:
        print("No images found.")
        return [], None

    dataset = ImageDataset(images, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=custom_collate_fn)

    scores = []
    for batch_images, _ in tqdm.tqdm(dataloader, desc="Processing Images"):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            image_features = model2.encode_image(batch_images)
            im_emb_arr = normalized(image_features.cpu().numpy())
            predictions = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
            scores.extend(predictions.cpu().numpy().flatten().tolist())

    if scores:
        average_score = sum(scores) / len(scores)
        print(f"Average aesthetic score for all images: {average_score}")
        return scores, average_score
    else:
        return scores, None

