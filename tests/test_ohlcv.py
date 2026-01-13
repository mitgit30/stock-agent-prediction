from src.data.ingestion import fetch_ohlcv

try:
    data = fetch_ohlcv("AMZN")
    print(data)
except Exception as e:
    print(f"Error {e}")


