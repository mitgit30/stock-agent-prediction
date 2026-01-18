
import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def check_endpoint(name, url):
    print(f"Checking {name} ({url})...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"✅ Success! Response: {response.json()}")
            return True
        elif response.status_code == 404:
            print(f"❌ Endpoint not found (404). Server might be running old code.")
            return False
        else:
            print(f"⚠️ Warning: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection refused. Is the server running on port 8000?")
        return False


def main():
    print("--- Redis Verification Script ---")
    
    # Determine the correct port
    ports = [8000, 8001]
    base_url = None
    
    for port in ports:
        url = f"http://localhost:{port}"
        print(f"Checking for API at {url}...")
        try:
            # Check a known existing endpoint (e.g., docs or root) to verify server presence
            # But here we ideally want to check if the NEW endpoints exist.
            # So let's check /api/redis/keys directly.
            response = requests.get(f"{url}/api/redis/keys", timeout=2)
            if response.status_code == 200:
                print(f"✅ Found updated API at {url}")
                base_url = url
                break
            elif response.status_code == 404:
                print(f"⚠️ Found API at {url}, but it returned 404 for new endpoints. (Old Code running)")
                # We might want to use this if we can't find the new one, but let's keep searching
            else:
                print(f"Found something at {url} (Status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            print(f"❌ No connection at {url}")

    if not base_url:
        print("\n❌ Could not find the updated API on port 8000 or 8001.")
        print("Please ensure 'uvicorn backend.main:app' is running and you have restarted it.")
        return

    print(f"\nUsing API at: {base_url}")

    # Check 1: Redis Keys (Debug Endpoint)
    print("\n1. Redis Connection Check...")
    if not check_endpoint("Redis Keys", f"{base_url}/api/redis/keys"):
        return

    # Check 2: Parent Prediction (Should populate cache)
    print("\n2. Triggering Parent Prediction (populates cache)...")
    check_endpoint("Parent Prediction", f"{base_url}/api/predict-parent")

    # Check 3: Verify Cache
    print("\n3. Verifying Redis Keys again...")
    check_endpoint("Redis Keys", f"{base_url}/api/redis/keys")

if __name__ == "__main__":
    main()

