import torch
from typing import List 

# Custom collate function for DataLoader
def collate_fn(items: List[dict]) -> torch.Tensor:
    images = [torch.Tensor(item['image']).unsqueeze(0) for item in items]
    labels = [item['label'] for item in items]  # Use label as an integer (class index)

    images = torch.cat(images)  # Stack images along the batch dimension
    labels = torch.tensor(labels, dtype=torch.long)  # Convert to tensor of class indices

    return images, labels


def preprocessing_label(label: int) -> torch.Tensor:
    new_label = torch.zeros(10)
    new_label[label] = 1

    return new_label