

import io
import base64
import pandas as pd
from PIL import Image
from minio import Minio
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

class CustomDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_type: str, label_column: str):
        client = Minio(
            "minio1.sedimark.work",
            access_key="super",
            secret_key="doopersecret"
        )

        result = client.get_object(dataset_path.split("/")[0], "/".join(dataset_path.split("/")[1:]))
        self.dataset = pd.read_csv(io.StringIO(result.data.decode('utf-8')))
        self.dataset_type = dataset_type
        self.label_column = label_column
        if dataset_type == "images":
            feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset_type == "images":
            string_image = base64.b64decode(self.dataset.iloc[idx]["image"].split(",")[-1])
            image = Image.open(io.BytesIO(string_image))
            return image, self.dataset.iloc[idx][self.label_column]
        else:
            return self.dataset.iloc[idx][[column for column in self.dataset.columns if column != self.label_column]], self.dataset.iloc[idx][self.label_column]

