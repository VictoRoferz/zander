# Zander – Kurz-Dokumentation zum Neu-Setup 


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
Nach **neu einloggen** oder **reboot:**

```bash
docker ps
docker compose version
```

### 2) Repository klonen & Git konfigurieren
```bash
cd ~
git clone <REPO_URL> zander
cd zander
git config --global user.name "USERNAME"
git config --global user.email "EMAIL"
```

- **GitLab Personal Access Token** erstellen (mit obiger Mail)
- Token wird für **HTTPS-Zugriff** benötigt

### 3) Setup-Script ausführen (beide Pis)
```bash
cd ~/zander
chmod +x setup.sh
./setup.sh
```

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

### 2) Remote auf HTTPS umstellen
HTTPS-URL aus der Gitlab Website kopieren und einfügen

Beispiel:
```bash
git remote set-url origin https://git-ce.rwth-aachen.de/wzl-iqs3/quality-insights/research-projects/edih/zander.git
```

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

- `Label Studio API Key`: *Legacy Token* des benutzten Label-Studio Accounts  
- **Kamera (Pi3)**:
  - `CAMERA_WIDTH`, `CAMERA_HEIGHT` (nur unterstützte Auflösungen)
  - `CAMERA_INDEX` (meist `0`)
  - `CAMERA_FPS` (niedrig halten, z.B. 2)
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

4. Profil-Icon → Account & Settings → Legacy Token

5. Legacy Token anzeigen oder erstellen

7. Token in `.env` oder Docker-Compose eintragen:
```text
LABEL_STUDIO_API_KEY=<TOKEN>
```

> Für Webhooks und Server-zu-Server-Kommunikation **nur Legacy Token verwenden**, da dieser nicht rotiert.

### Label Studio Webhook (Pi5)

Project-Settings → Webhooks → Add Webhook

Webhook **in der Label-Studio-UI** konfigurieren:
```text
http://<PI5_IP>:8002/api/v1/webhook/annotation-created
```

---

## Arbeiten im Docker-Container (Bash öffnen)

### Laufende Container anzeigen
```bash
docker ps
```



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

## Kamera Setup (Pi3)

### Voraussetzungen

- Basler GigE-Vision-Kamera ist per **LAN / PoE** angeschlossen  
- Kamera und Raspberry Pi befinden sich im **gleichen Netzwerk**
- pypylon ist im docker installiert (https://github.com/basler/pypylon)
- Docker services sind auf `network_mode: "host"` gesetzt
  

### Kamera im Docker finden

Sobald der Kamera-Container läuft, kann geprüft werden, ob pylon die Kamera sieht:

```bash
docker exec -it pcb-server1-camera \
  python3 -c "from pypylon import pylon; print(pylon.TlFactory.GetInstance().EnumerateDevices())"
```

#### Erwartetes Ergebnis

- Mindestens ein Eintrag (z. B. ein DeviceInfo-Objekt) → Kamera ist gefunden und erreichbar  
- Leeres Ergebnis `()` → Kamera wird nicht gefunden


#### Wenn das Ergebnis leer ist `()`:


- Kamera hat keinen Strom (Licht an der Oberseite leuchtet nicht)
- Kamera ist nicht im gleichen Netz wie der Raspberry Pi
- `network_mode: host` fehlt in einem docker service
- Ethernet-Switch blockiert Broadcasts
 

### Datenübertragungsproblem

**Fehlermeldung:**
- `The buffer was incompletely grabbed`

**Ursache:**
- Raspberry Pi 3 hat nur **100 Mbit/s Ethernet**
- Hardware (Ethernetkabel, Switches) können auch Bandbreite begrenzen
- 5 MP GigE-Kamera kann die verfügbare Bandbreite überlasten 

**Lösung im Code (`_init_pylon` in `camera_service.py`):**

- `DeviceLinkThroughputLimit` auf z.B. **40 Mbit/s** begrenzen

**Einstellungen der Kamera ändern (in ENV oder docker-compose)**

- FPS reduzieren (z. B. 2 fps)
- Auflösung reduzieren (z. B. 1920×1080)
---

## Button-Setup & Pin-Verdrahtung (Pi3)

### Verdrahtung
- Button-Kontakt 1 → **GND**
- Button-Kontakt 2 → **physikalischer Pin 16**
- Physikalischer Pin 16 entspricht **GPIO23**, wird der Button an einen anderen Pin angeschlossen muss `server1_camera/scripts/button_listener.py`  entsprechend geändert werden
- Referenz: https://digitalewelt.at/raspberry-pi-taster-abfragen/

---

## Daten & Volumes prüfen (Pi5)


```bash
docker exec -it pcb-server2-labelstudio ls -l /data/unlabeled/
docker exec -it pcb-server2-labelstudio ls -l /data/labeled/
```

---

## Remote-Zugriff (SSH / remote.it)

- SSH-Keys des Client-Geräts in:
```bash
~/.ssh/authorized_keys
```
- Verbinden über die Konsole des Client-Geräts:
```bash
ssh pi@<Pi-IP>
```
---

## Sammlung von Setup/Debug-Befehlen

```bash
docker compose -f docker-compose.pi3.yml up -d --build
docker compose -f docker-compose.pi5.yml up -d --build
docker compose -f docker-compose.pi3.yml down
docker compose -f docker-compose.pi5.yml down
docker compose -f docker-compose.pi3.yml logs -f
docker compose -f docker-compose.pi5.yml logs -f
docker ps
docker exec -it <container_name> bash
pinctrl get 23
Sudo nmtui
Sudo iwlist wlan0 scan
hostname -I
docker exec -it pcb-server1-camera \
  python3 -c "from pypylon import pylon; print(pylon.TlFactory.GetInstance().EnumerateDevices())"
curl -X POST "http://<PI5_IP>:8001/api/v1/button-capture"
```
