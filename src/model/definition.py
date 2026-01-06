import torch
import torch.nn as nn
from src.config import Config

# Define the LSTM based model for stock price prediction
class LSTMModel(nn.Module):
    """LSTM Model for Stock Price Prediction."""
    def __init__(self,input_size: int = Config().input_size, hidden_size: int = 128, num_layers: int = 3, pred_len: int = Config().pred_len,dropout:float=0.2):
        
        super().__init__() # initialize and bound to parent class nn.Module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size * pred_len) # fully connected layer for predict the entire future in one prediction shot
        self.pred_len = pred_len
        self.input_size = input_size
    
    #  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # defines how data should be flow theough the LSTM model
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        out = out.view(batch_size, self.pred_len, self.input_size) # returns the output reshaped to (batch_size, pred_len, input_size)
        return out
    
    # refer LSTM model more at https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html