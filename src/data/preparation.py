import torch
from torch.utils.data import Dataset
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from src.config import Config
from src.exception import PipelineError
import pandas as pd

class StockDataset(Dataset):
    """Dataset for stock price sequences (Data Preparation).
    
    
    """
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler, context_len: int = Config().context_len, pred_len: int = Config().pred_len):
        self.context_len = context_len
        self.pred_len = pred_len
        try:
            vals = scaler.transform(df[Config().features]).astype("float32")
            self.samples = []
            
            # this for loop converts normalized stock features into rolling time-series sequences so a PyTorch model can learn to predict future prices from past behavior.
            # from each time step t, it takes the previous context_len steps as input (past) and the next pred_len steps as target (fut).
            # the prediction will of 5 days into the future based on the past 60 days of data.
            for t in range(context_len, len(df) - pred_len):
                past = vals[t - context_len:t]
                fut = vals[t:t + pred_len]
                if past.shape == (context_len, len(Config().features)) and fut.shape == (pred_len, len(Config().features)):
                    self.samples.append((past, fut))
                else:
                    print(f"Skipping invalid sample at index {t}: past shape {past.shape}, fut shape {fut.shape}")
            if not self.samples:
                raise PipelineError("No valid samples created for dataset")
        except Exception as e:
            raise PipelineError(f"Failed to create dataset: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        past, fut = self.samples[idx]
        return torch.tensor(past), torch.tensor(fut)