from __future__ import annotations

import importlib
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

HOST="127.0.0.1"
PORT=8000
HEALTH=f"http://{HOST}:{PORT}/health"
UI=f"http://{HOST}:{PORT}/ui/"


def wait_and_open():
    for _ in range(120):
        try:
            with urllib.request.urlopen(HEALTH,timeout=1) as r:
                if r.status==200:
                    print(f"\n[OK] Backend saludable: {HEALTH}")
                    print(f"[OK] Abriendo: {UI}\n")
                    webbrowser.open(UI)
                    return
        except Exception:
            time.sleep(1)
    print("[ERROR] El backend no respondio al health check en 120 s.")


def check_port():
    with socket.socket() as s:
        return s.connect_ex((HOST,PORT))==0


def main():
    print("="*50)
    print(" APEND DETECTION 1.1 - LOCALHOST")
    print("="*50)
    print("Python:",sys.version.split()[0])
    if sys.version_info < (3,10):
        raise SystemExit("[ERROR] Se requiere Python 3.10 o superior.")

    required=["fastapi","uvicorn","pandas","numpy","geopandas","scipy","sklearn","shapely","pyproj"]
    missing=[]
    for name in required:
        try: importlib.import_module(name)
        except Exception as e:
            missing.append(f"{name}: {e}")
    if missing:
        print("[ERROR] Dependencias que no cargan:")
        for x in missing: print(" -",x)
        raise SystemExit(1)

    try:
        from geo_outliers.api import app
        print("[OK] Backend importado:",app.title,app.version)
    except Exception:
        print("\n[ERROR] Fallo importando geo_outliers.api:\n")
        raise

    if check_port():
        print(f"[INFO] El puerto {PORT} ya esta ocupado. Abriendo la interfaz existente.")
        webbrowser.open(UI)
        return

    threading.Thread(target=wait_and_open,daemon=True).start()
    print(f"[START] http://{HOST}:{PORT}")
    print("[INFO] Mantenga esta ventana abierta. CTRL+C para detener.\n")
    import uvicorn
    uvicorn.run(app,host=HOST,port=PORT,log_level="info")


if __name__=="__main__":
    main()
