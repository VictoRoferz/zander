# server1_camera/scripts/button_listener.py

import os
import time
import requests
import RPi.GPIO as GPIO

BUTTON_PIN = 23  # BCM 23 = physischer Pin 16

SERVER1_URL = os.getenv("SERVER1_URL", "http://pcb-server1-camera:8001")
BUTTON_ENDPOINT = f"{SERVER1_URL}/api/v1/button"


def main():
    print(f"[Button Listener] Starte GPIO Listener auf BCM {BUTTON_PIN}")
    print(f"[Button Listener] Trigger-URL: {BUTTON_ENDPOINT}")

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    last_state = GPIO.input(BUTTON_PIN)

    try:
        while True:
            state = GPIO.input(BUTTON_PIN)

            # Button hat seinen Zustand geändert:
            if state != last_state:
                if state == GPIO.LOW:
                    print("[Button Listener] ➡ Button GEDRÜCKT – sende POST...")

                    try:
                        resp = requests.get(BUTTON_ENDPOINT, timeout=5)
                        print(f"[Button Listener] Antwort: "
                              f"{resp.status_code} | {resp.text}")
                    except Exception as e:
                        print(f"[Button Listener] Fehler beim Request: {e}")

                else:
                    print("[Button Listener] ⬅ Button LOSGELASSEN")

                last_state = state
                time.sleep(0.05)  # Entprellen

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[Button Listener] Stop durch KeyboardInterrupt")

    finally:
        GPIO.cleanup()
        print("[Button Listener] GPIO.cleanup() ausgeführt")


if __name__ == "__main__":
    main()
