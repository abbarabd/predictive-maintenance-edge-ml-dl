import paho.mqtt.client as mqtt
import json
import csv
import datetime
import sys
import signal
import os

# --- Configuration MQTT ---
MQTT_BROKER = "127.0.0.1"  # Mosquitto tourne sur le RPi lui-même
MQTT_PORT = 1883           # Port MQTT standard
MQTT_TOPIC = "machine/moteur1/data" # Doit correspondre au topic de l'ESP32
CSV_FILENAME = "sensor_data_collection.csv" # Nom du fichier CSV pour la sauvegarde

# --- Variables Globales ---
sample_count = 0        # Compteur d'échantillons collectés
csv_file_handle = None  # Handle du fichier CSV
csv_writer = None       # Objet writer pour CSV
headers_written = False # Indique si l'en-tête du CSV a été écrit
# Fréquence d'échantillonnage de l'ESP32. DOIT CORRESPONDRE AU CODE ESP32.
ESP32_READ_INTERVAL_MS = 100 

# --- Fonction de gestion d'arrêt propre (Ctrl+C) ---
def signal_handler(sig, frame):
    """
    Gère le signal SIGINT (Ctrl+C) pour arrêter proprement la collecte.
    Ferme le fichier CSV avant de quitter.
    """
    print(f"\nSignal {sig} reçu. Arrêt de la collecte de données.")
    if csv_file_handle and not csv_file_handle.closed:
        csv_file_handle.close()
        print(f"Fichier CSV '{CSV_FILENAME}' fermé avec succès.")
    sys.exit(0)

# Enregistrer le gestionnaire de signal pour SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# --- Fonctions de Callback MQTT ---
def on_connect(client, userdata, flags, rc):
    """
    Callback appelé lorsque le client MQTT se connecte au broker.
    Abonne le client au topic des données.
    """
    if rc == 0:
        print(f"Connected to MQTT Broker ({MQTT_BROKER}:{MQTT_PORT})!")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to topic: '{MQTT_TOPIC}'")
    else:
        print(f"Failed to connect, return code {rc}\n")
        sys.exit(1)

def on_message(client, userdata, msg):
    """
    Callback appelé lorsque le client reçoit un message sur un topic abonné.
    Décode le message, ajoute un timestamp, et l'écrit dans le fichier CSV.
    """
    global sample_count, headers_written, csv_writer, csv_file_handle

    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)

        # Ajouter un timestamp local du Raspberry Pi
        data['timestamp_rpi'] = datetime.datetime.now().isoformat(timespec='milliseconds')

        # Vérifier si l'en-tête du CSV a déjà été écrit
        if not headers_written:
            headers = list(data.keys())
            csv_writer.writerow(headers)
            headers_written = True

        csv_writer.writerow(list(data.values()))
        
        sample_count += 1

        if sample_count % 10 == 0: # Afficher toutes les 10 lectures (1 seconde de données)
            print(f"\rSamples collected: {sample_count}", end="", flush=True)

    except json.JSONDecodeError:
        print(f"Error decoding JSON: {msg.payload.decode('utf-8')}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Fonction principale de démarrage de la collecte ---
def main():
    global csv_file_handle, csv_writer, headers_written

    try:
        # Vérifier si le fichier existe et s'il est vide pour écrire l'en-tête
        file_exists = os.path.exists(CSV_FILENAME)
        file_is_empty = not file_exists or os.path.getsize(CSV_FILENAME) == 0

        csv_file_handle = open(CSV_FILENAME, 'a', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file_handle)

        if file_is_empty:
            # L'en-tête sera écrit lors du premier message dans on_message
            headers_written = False
        else:
            headers_written = True # Si le fichier n'est pas vide, on suppose que l'en-tête est déjà là

        print(f"Starting data collection into '{CSV_FILENAME}'...")
        print("Press Ctrl+C to stop the collection.")

        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message

        client.connect(MQTT_BROKER, MQTT_PORT, 60)

        client.loop_forever()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred during MQTT connection or loop: {e}")
    finally:
        if csv_file_handle and not csv_file_handle.closed:
            csv_file_handle.close()
            print(f"\nFinalisation: Fichier CSV '{CSV_FILENAME}' fermé.")

if __name__ == "__main__":
    main()
