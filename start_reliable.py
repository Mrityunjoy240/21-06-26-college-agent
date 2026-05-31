import subprocess
import time
import requests
import os
import signal
import sys

def run_command(command, name, cwd=None):
    print(f"Starting {name}...")
    # Use CREATE_NEW_PROCESS_GROUP on Windows to avoid signal sharing
    return subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        cwd=cwd
    )

def main():
    print("=== BCREC Voice Agent Reliable Starter ===")
    
    # 1. Kill existing
    if sys.platform == "win32":
        subprocess.run("taskkill /F /IM ngrok.exe /T", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.run("taskkill /F /IM python.exe /T", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    # 2. Start Ngrok
    ngrok = run_command("ngrok http 8000", "Ngrok")
    time.sleep(3)

    # 3. Start Backend
    backend = run_command("python -m uvicorn app.main:app --host 0.0.0.0 --port 8000", "Backend", cwd="backend")
    
    # 4. Start Agent
    agent = run_command("python scripts/livekit_agent.py direct", "LiveKit Agent")

    print("\nWaiting for services to initialize...")
    ready = False
    for i in range(20):
        try:
            res = requests.get("http://localhost:8000/health", timeout=1)
            if res.status_code == 200:
                ready = True
                break
        except:
            pass
        time.sleep(1)
        print(".", end="", flush=True)

    if ready:
        print("\n\n✅ ALL SERVICES READY!")
        print("You can now run: python trigger_call.py <phone_number>")
        print("Keep this script running. Press Ctrl+C to stop all services.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            ngrok.terminate()
            backend.terminate()
            agent.terminate()
    else:
        print("\n\n❌ Timed out waiting for backend. Check backend/logs/app.log")
        ngrok.terminate()
        backend.terminate()
        agent.terminate()

if __name__ == "__main__":
    main()
