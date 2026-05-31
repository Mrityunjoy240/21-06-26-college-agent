import requests
import json
import time
import sys

def get_ngrok_url():
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        tunnels = response.json().get('tunnels', [])
        for tunnel in tunnels:
            if tunnel.get('proto') == 'https':
                return tunnel.get('public_url')
    except:
        pass
    return None

def trigger(phone_number):
    base_url = get_ngrok_url()
    if not base_url:
        print("Error: Ngrok not running or API (4040) not accessible.")
        return

    url = "http://localhost:8000/voice/call-me"
    payload = {
        "phone_number": phone_number,
        "base_url": base_url
    }

    print(f"Triggering call to {phone_number} via {base_url}...")
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    number = sys.argv[1] if len(sys.argv) > 1 else "+919641089749"
    trigger(number)
