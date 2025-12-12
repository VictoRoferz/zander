# Zander – Kurz-Dokumentation zum Neu-Setup (2× Raspberry Pi)

---

## Allgemeine Hinweise
- **Pi3**: Kamera + Button → Bild aufnehmen → Upload zu Pi5  
- **Pi5**: Server + Storage (Docker Volume) → Label Studio Tasks → Training & Inferenz  
- Zugriff & Wartung über **SSH / remote.it**  
- Alle Services laufen in Docker-Containern

---

## Setup-Ablauf

### 1) Basisinstallation (beide Pis)
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```
### neu einloggen oder reboot
```bash
docker ps
docker compose version
```

---

### 2) Repository klonen & Git konfigurieren
```bash
cd ~
git clone <REPO_URL> zander
cd zander
git config --global user.name "USERNAME"
git config --global user.email "EMAIL"
```

- **GitLab Personal Access Token** erstellen (mit obiger Mail)
- Token wird für **HTTPS-Zugriff** benötigt (empfohlen auf Raspberry Pis)

---

### 3) Setup-Script ausführen (beide Pis)
```bash
cd ~/zander
chmod +x setup.sh
./setup.sh
```

---

### 4) Services starten

**Pi3 (Camera Service)**
```bash
cd ~/zander/backend
docker compose -f docker-compose.pi3.yml up -d --build
```

**Pi5 (Server + Label Studio)**
```bash
cd ~/zander/backend
docker compose -f docker-compose.pi5.yml up -d --build
```

**Stoppen**
```bash
docker compose -f docker-compose.pi3.yml down
docker compose -f docker-compose.pi5.yml down
```

---

## Repository von SSH auf HTTPS umstellen

Falls das Repository **initial per SSH** geklont oder gepusht wurde, aber nun
**HTTPS mit Personal Access Token** genutzt werden soll:

### 1) Aktuelle Remote-URL prüfen
```bash
git remote -v
```

Beispiel (SSH):
```text
origin  git@git-ce.rwth-aachen.de:wzl-iqs3/quality-insights/research-projects/edih/zander.git (fetch)
origin  git@git-ce.rwth-aachen.de:wzl-iqs3/quality-insights/research-projects/edih/zander.git (push)
```

---

### 2) Remote auf HTTPS umstellen
HTTPS-URL aus der Gitlab Website kopieren und einfügen

Beispiel:
```bash
git remote set-url origin https://git-ce.rwth-aachen.de/wzl-iqs3/quality-insights/research-projects/edih/zander.git
```

---

### 3) Test (Pull oder Push)
```bash
git pull
```

- **Username**: GitLab Benutzername  
- **Password**: **Personal Access Token** (nicht das GitLab-Passwort)


---

## Wichtige Konfigurationen (ENV)

- `SERVER2_URL`: IP des **Pi5**
```bash
hostname -I
```

- **Label Studio API Key**: *Legacy Token* des Accounts  
- **Kamera (Pi3)**:
  - `CAMERA_WIDTH`, `CAMERA_HEIGHT` (nur unterstützte Auflösungen)
  - `CAMERA_INDEX` (meist `0`)

---

## Label Studio: Legacy API Token finden & aktivieren (Pi5)

1. Label Studio UI öffnen:
```text
http://<PI5_IP>:8080
```
2. Anmelden
```text
labelling@zander-aachen.de
zander123
```
3. Dropdown Menü öffnen → Organization → API Token Settings → Legacy Token aktivieren

4. Profil-Icon → **Account & Settings** → Legacy Token

5. **Legacy Token anzeigen oder erstellen**

6. Token in `.env` oder Docker-Compose eintragen:
```text
LABEL_STUDIO_API_KEY=<TOKEN>
```

> Für Webhooks und Server-zu-Server-Kommunikation **nur Legacy Token verwenden**, da dieser nicht rotiert.

---

## Arbeiten im Docker-Container (Bash öffnen)

### Laufende Container anzeigen
```bash
docker ps
```

---

### Bash im Container starten
```bash
docker exec -it <container_name> bash
```

Beispiel:
```bash
docker exec -it pcb-server1-camera bash
```

Beenden:
```bash
exit
```

---

## Kamera-Debug (Pi3)

### Kamera-Gerät prüfen (im Container)
```bash
ls /dev/video*
```

---

### OpenCV-Capture-Test (im `server1`-Container)
```bash
python - << 'EOF'
import cv2
idx = 0
cap = cv2.VideoCapture(idx)
ok, frame = cap.read()
print("Index:", idx, "opened:", cap.isOpened(), "frame:", ok, "shape:", getattr(frame, "shape", None))
cap.release()
EOF
```

**Erwartet:**  
- `opened: True`  
- `frame: True`  

---

## Button-Setup & Pin-Verdrahtung (Pi3)

### Verdrahtung
- Button-Kontakt 1 → **GND**
- Button-Kontakt 2 → **physikalischer Pin 16**
- Physikalischer Pin 16 entspricht **GPIO23**
- Referenz: https://digitalewelt.at/raspberry-pi-taster-abfragen/

## Label Studio Webhook (Pi5)

Project-Settings → Webhooks → Add Webhook

Webhook **in der Label-Studio-UI** konfigurieren:
```text
http://<PI5_IP>:8002/api/v1/webhook/annotation-created
```

---

## Daten & Volumes prüfen (Pi5)


```bash
docker exec -it pcb-server2-labelstudio ls -l /data/unlabeled/
docker exec -it pcb-server2-labelstudio ls -l /data/labeled/
```

---

## Sammlung von Debug-Befehlen

```bash
docker compose -f docker-compose.pi3.yml logs -f
docker compose -f docker-compose.pi5.yml logs -f
docker ps
docker exec -it <container_name> bash
ls /dev/video*
pinctrl get 23
Sudo nmtui
Sudo iwlist wlan0 scan 
```

---

## Remote-Zugriff (SSH / remote.it)

- SSH-Keys des Client-Geräts in:
```bash
~/.ssh/authorized_keys
```
```bash
ssh pi@<Pi-IP>
```