import torch

class synthetic_dataset():
    """Synthetic two-dimensional regression task"""
    
    def __init__(self, n_obs):
        self.n_observations = n_obs
        x = torch.arange(start=0, end=1, step=1/n_obs)
        self.features = torch.cat((x, torch.cos(x))).reshape(n_obs, 2)
        self.target = (self.features[:, 0] + self.features[:, 1] + 
                       torch.rand(n_obs))
        self.dataset = (self.features, self.target)
        
    def __len__(self):
        return self.n_observations
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.features[idx]
        target = self.target[idx]
        return data, target