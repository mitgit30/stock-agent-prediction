## Helper function for predicting next 5 days forcast using pytorch model

from logger.logger import get_logger
from sklearn.preprocessing import StandardScaler
from typing import Dict
from src.config import Config
import numpy as np
import pandas as pd
from src.exception import PipelineError
import torch


logger = get_logger()   

def predict_one_step_and_week(model, df: pd.DataFrame, scaler: StandardScaler, ticker: str) -> Dict:
    """Predict next 5 days forcast using pytorch model and return the predicted values , it will predict next day, next week and full 5 day forcast."""
    
    try:
        config = Config()
        vals = scaler.transform(df[config.features]).astype("float32")

        X = vals[-config.context_len:].reshape(1, config.context_len, config.input_size)
        
        with torch.no_grad(): # Run the inference loop
            x_tensor = torch.tensor(X, dtype=torch.float32).to(config.device)
            pred = model(x_tensor).cpu().numpy()[0]
        
        pred_inverse = scaler.inverse_transform(pred.reshape(-1,config.input_size,))[:,:5]
        
        ## Prepare dates
        last_date=df["date"].iloc[-1]
        next_days = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=config.pred_len)
        
         # Construct forecasts
        forecast = []
        for i, date in enumerate(next_days):
            forecast.append({
                "date": str(date.date()),
                "open": float(pred_inverse[i][0]),
                "high": float(pred_inverse[i][1]),
                "low": float(pred_inverse[i][2]),
                "close": float(pred_inverse[i][3]),
                "volume": float(pred_inverse[i][4])
            })

        # Response structure
        return {
            "ticker": ticker,
            "last_date": str(last_date.date()),
            "future_window_days": config.pred_len,
            "next_business_days": [str(d.date()) for d in next_days],
            "predictions": {
                "next_day": forecast[0],
                "next_week": {
                    "high": float(np.max([d["high"] for d in forecast])),
                    "low": float(np.min([d["low"] for d in forecast]))
                },
                "full_forecast": forecast
            }
        }
        
    except Exception as e:
        raise PipelineError(f"Error predicting next 5 days forcast: {e}")
        }

        