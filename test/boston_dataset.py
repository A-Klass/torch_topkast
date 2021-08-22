from sklearn import datasets
import torch

class boston_dataset(Dataset):
    """Boston dataset."""

    def __init__(self):
        self.dataset = datasets.load_boston()

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = torch.from_numpy(self.dataset.data[idx])
        target = torch.tensor(self.dataset.target[idx])
        return data, target