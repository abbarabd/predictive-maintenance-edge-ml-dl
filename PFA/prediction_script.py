import paho.mqtt.client as mqtt
import json
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import joblib # Pour charger les modèles et outils
import datetime
import time
import collections # Pour le deque, un type de liste optimisé pour les buffers
import sys
import signal
import tensorflow as tf # Pour charger le modèle Keras/TF
import requests # Pour les requêtes HTTP de fallback
import threading # Pour les tâches en arrière-plan

# --- Configuration MQTT ---
MQTT_BROKER = "127.0.0.1"  # Mosquitto tourne sur le RPi lui-même
MQTT_PORT = 1883
MACHINE_ID = "moteur3"  # ID unique pour ce moteur 

# TOPICS MQTT COMPATIBLES AVEC LE SERVEUR NODE.JS
MQTT_TOPIC_DATA_IN = f"machine/{MACHINE_ID}/data"  # Topic pour les données brutes de l'ESP32
MQTT_TOPIC_SENSOR_OUT = f"sensor/{MACHINE_ID}/data"  # Topic pour envoyer les données au serveur
MQTT_TOPIC_PREDICTION_OUT = f"prediction/{MACHINE_ID}/alert"  # Topic pour les prédictions
MQTT_TOPIC_METRICS_OUT = f"metrics/{MACHINE_ID}/update"  # Topic pour les métriques
MQTT_TOPIC_DEVICE_STATUS = f"device/{MACHINE_ID}/status"  # Topic pour le statut

# --- Configuration du Serveur Backend ---
BACKEND_SERVER_URL = "http://localhost:3001"  # URL du serveur Node.js
HTTP_FALLBACK_ENABLED = True  # Utiliser HTTP si MQTT échoue

# --- Configuration du Modèle et des Caractéristiques ---
XGB_MODEL_PATH = "xgboost_motor_model.pkl"
CNN_LSTM_MODEL_PATH = "cnn_lstm_motor_model.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
XGB_FEATURE_COLUMNS_PATH = "xgb_feature_columns.pkl"
DL_RAW_DATA_COLUMNS_PATH = "dl_raw_data_columns.pkl"
DL_SCALER_PATH = "dl_scaler.pkl"

# Taille de la fenêtre en secondes (DOIT ÊTRE LA MÊME QUE LORS DE L'ENTRAÎNEMENT !)
WINDOW_SIZE_SECONDS = 1 
# Intervalle de lecture de l'ESP32 en ms (DOIT CORRESPONDRE À VOTRE CODE ESP32)
ESP32_READ_INTERVAL_MS = 100 
# Nombre d'échantillons attendus par fenêtre
EXPECTED_SAMPLES_PER_WINDOW = int(WINDOW_SIZE_SECONDS * (1000 / ESP32_READ_INTERVAL_MS))

# --- Variables Globales ---
data_buffer = collections.deque(maxlen=EXPECTED_SAMPLES_PER_WINDOW)
last_prediction_time = time.time()
last_metrics_time = time.time()
PREDICTION_INTERVAL_SECONDS = 1 # Fréquence à laquelle les prédictions sont effectuées
METRICS_INTERVAL_SECONDS = 30 # Envoyer les métriques toutes les 30 secondes
STATUS_INTERVAL_SECONDS = 60 # Envoyer le statut toutes les minutes

# Variables pour les métriques
total_predictions = 0
anomaly_count = 0
last_sensor_data = None
mqtt_client = None

# Charger les modèles et outils (une seule fois au démarrage)
try:
    loaded_xgb_model = joblib.load(XGB_MODEL_PATH)
    loaded_cnn_lstm_model = tf.keras.models.load_model(CNN_LSTM_MODEL_PATH)
    loaded_label_encoder = joblib.load(LABEL_ENCODER_PATH)
    loaded_xgb_feature_columns = joblib.load(XGB_FEATURE_COLUMNS_PATH)
    loaded_dl_raw_data_columns = joblib.load(DL_RAW_DATA_COLUMNS_PATH)
    loaded_dl_scaler = joblib.load(DL_SCALER_PATH)

    print(f"✅ Modèles et outils chargés avec succès.")
    print("📋 Classes reconnues par le modèle:", loaded_label_encoder.classes_)
except FileNotFoundError as e:
    print(f"❌ Erreur: Fichier essentiel non trouvé: {e}.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles/outils: {e}")
    sys.exit(1)

# --- Fonction d'envoi HTTP de fallback ---
def send_http_fallback(endpoint, data):
    """Envoie des données via HTTP si MQTT échoue"""
    if not HTTP_FALLBACK_ENABLED:
        return False
    
    try:
        url = f"{BACKEND_SERVER_URL}/{endpoint}"
        response = requests.post(url, json=data, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"⚠️ HTTP fallback échoué: {e}")
        return False

# --- Fonction d'envoi de données capteur au serveur ---
def send_sensor_data_to_server(sensor_data):
    """Envoie les données capteur au serveur Node.js"""
    # Format attendu par le serveur
    formatted_data = {
        "machine_id": MACHINE_ID,
        "timestamp_rpi": sensor_data.get("timestamp", datetime.datetime.now().isoformat()),
        "temperature_c": sensor_data.get("temperature_c"),
        "sound_amplitude": sensor_data.get("sound_amplitude"),
        "accel_x_g": sensor_data.get("accel_x_g"),
        "accel_y_g": sensor_data.get("accel_y_g"),
        "accel_z_g": sensor_data.get("accel_z_g")
    }
    
    # Essayer MQTT d'abord
    try:
        if mqtt_client:
            mqtt_client.publish(MQTT_TOPIC_SENSOR_OUT, json.dumps(formatted_data))
            return True
    except Exception as e:
        print(f"⚠️ Échec MQTT pour données capteur: {e}")
    
    # Fallback HTTP
    return send_http_fallback("api/raspberry-pi/sensor-data", formatted_data)

# --- Fonction d'envoi de métriques ---
def send_metrics_update():
    """Envoie les métriques au serveur"""
    global last_sensor_data
    
    if not last_sensor_data:
        return
    
    metrics = {
        "total_predictions": total_predictions,
        "anomaly_count": anomaly_count,
        "anomaly_rate": (anomaly_count / max(total_predictions, 1)) * 100,
        "current_temperature": last_sensor_data.get("temperature_c"),
        "current_vibration": np.sqrt(
            (last_sensor_data.get("accel_x_g", 0)**2 + 
             last_sensor_data.get("accel_y_g", 0)**2 + 
             last_sensor_data.get("accel_z_g", 0)**2)
        ),
        "current_sound": last_sensor_data.get("sound_amplitude"),
        "uptime": time.time() - start_time,
        "buffer_size": len(data_buffer),
        "last_updated": datetime.datetime.now().isoformat()
    }
    
    # Essayer MQTT d'abord
    try:
        if mqtt_client:
            mqtt_client.publish(MQTT_TOPIC_METRICS_OUT, json.dumps(metrics))
            return True
    except Exception as e:
        print(f"⚠️ Échec MQTT pour métriques: {e}")
    
    return False

# --- Fonction d'envoi de statut ---
def send_device_status():
    """Envoie le statut du device"""
    status = {
        "device_id": MACHINE_ID,
        "status": "online",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0.0",
        "models_loaded": True,
        "buffer_capacity": EXPECTED_SAMPLES_PER_WINDOW,
        "prediction_interval": PREDICTION_INTERVAL_SECONDS
    }
    
    # Essayer MQTT
    try:
        if mqtt_client:
            mqtt_client.publish(MQTT_TOPIC_DEVICE_STATUS, json.dumps(status))
            return True
    except Exception as e:
        print(f"⚠️ Échec MQTT pour statut: {e}")
    
    return False

# --- Fonction d'extraction des caractéristiques pour XGBoost (identique à l'entraînement) ---
def extract_xgb_features_from_window(window_df):
    # Traitement des NaN comme lors de l'entraînement
    window_df_copy = window_df.copy()
    window_df_copy['temperature_c'] = window_df_copy['temperature_c'].replace(-1.0, np.nan)
    window_df_copy = window_df_copy.fillna(method='ffill').fillna(method='bfill').fillna(0)

    features = {
        'temp_mean': window_df_copy['temperature_c'].mean(),
        'temp_std': window_df_copy['temperature_c'].std(),
        'temp_min': window_df_copy['temperature_c'].min(),
        'temp_max': window_df_copy['temperature_c'].max(),

        'sound_rms_mean': window_df_copy['sound_amplitude'].mean(),
        'sound_rms_std': window_df_copy['sound_amplitude'].std(),
        'sound_rms_max': window_df_copy['sound_amplitude'].max(),

        'accel_x_rms': np.sqrt(np.mean(window_df_copy['accel_x_g']**2)),
        'accel_y_rms': np.sqrt(np.mean(window_df_copy['accel_y_g']**2)),
        'accel_z_rms': np.sqrt(np.mean(window_df_copy['accel_z_g']**2)),
        
        'accel_x_std': window_df_copy['accel_x_g'].std(),
        'accel_y_std': window_df_copy['accel_y_g'].std(),
        'accel_z_std': window_df_copy['accel_z_g'].std(),

        'accel_x_ptp': window_df_copy['accel_x_g'].max() - window_df_copy['accel_x_g'].min(),
        'accel_y_ptp': window_df_copy['accel_y_g'].max() - window_df_copy['accel_y_g'].min(),
        'accel_z_ptp': window_df_copy['accel_z_g'].max() - window_df_copy['accel_z_g'].min(),

        'accel_x_skew': skew(window_df_copy['accel_x_g']),
        'accel_y_skew': skew(window_df_copy['accel_y_g']),
        'accel_z_skew': skew(window_df_copy['accel_z_g']),

        'accel_x_kurt': kurtosis(window_df_copy['accel_x_g']),
        'accel_y_kurt': kurtosis(window_df_copy['accel_y_g']),
        'accel_z_kurt': kurtosis(window_df_copy['accel_z_g']),
    }
    
    for key, value in features.items():
        if isinstance(value, float) and np.isnan(value):
            features[key] = 0.0

    return pd.DataFrame([features])

# --- Callbacks MQTT ---
def on_connect(client, userdata, flags, rc):
    global mqtt_client
    if rc == 0:
        print(f"✅ Connecté au broker MQTT ({MQTT_BROKER}:{MQTT_PORT})!")
        mqtt_client = client
        client.subscribe(MQTT_TOPIC_DATA_IN)
        print(f"📡 Abonné au topic: '{MQTT_TOPIC_DATA_IN}'")
        
        # Envoyer le statut de connexion
        send_device_status()
    else:
        print(f"❌ Échec de la connexion MQTT, code: {rc}")
        if not HTTP_FALLBACK_ENABLED:
            sys.exit(1)

def on_message(client, userdata, msg):
    global data_buffer, last_prediction_time, last_sensor_data, total_predictions, anomaly_count

    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        last_sensor_data = data

        # Envoyer les données capteur au serveur
        send_sensor_data_to_server(data)

        data_to_buffer = {k: v for k, v in data.items() if k in loaded_dl_raw_data_columns}
        data_buffer.append(data_to_buffer)

        if len(data_buffer) >= EXPECTED_SAMPLES_PER_WINDOW and (time.time() - last_prediction_time) >= PREDICTION_INTERVAL_SECONDS:
            last_prediction_time = time.time()
            total_predictions += 1

            current_window_df = pd.DataFrame(list(data_buffer))

            # --- Prédiction avec XGBoost ---
            xgb_features = extract_xgb_features_from_window(current_window_df)
            xgb_features = xgb_features[loaded_xgb_feature_columns]
            xgb_prediction_encoded = loaded_xgb_model.predict(xgb_features)
            xgb_prediction_label = loaded_label_encoder.inverse_transform(xgb_prediction_encoded)[0]
            xgb_confidence = loaded_xgb_model.predict_proba(xgb_features)[0][xgb_prediction_encoded[0]]

            # --- Prédiction avec CNN-LSTM ---
            dl_raw_data = current_window_df[loaded_dl_raw_data_columns].values
            dl_raw_data_scaled = loaded_dl_scaler.transform(dl_raw_data.reshape(-1, dl_raw_data.shape[-1]))
            dl_input = dl_raw_data_scaled.reshape(1, EXPECTED_SAMPLES_PER_WINDOW, len(loaded_dl_raw_data_columns))
            
            dl_prediction_proba = loaded_cnn_lstm_model.predict(dl_input, verbose=0)[0]
            dl_prediction_encoded = np.argmax(dl_prediction_proba)
            dl_prediction_label = loaded_label_encoder.inverse_transform([dl_prediction_encoded])[0]
            dl_confidence = dl_prediction_proba[dl_prediction_encoded]

            # --- Logique de Fusion des Décisions ---
            final_prediction_label = "Normal"
            final_confidence = 0.0
            
            if xgb_prediction_label != 'Normal' and dl_prediction_label != 'Normal':
                if xgb_confidence > dl_confidence:
                    final_prediction_label = xgb_prediction_label
                    final_confidence = xgb_confidence
                else:
                    final_prediction_label = dl_prediction_label
                    final_confidence = dl_confidence
            elif xgb_prediction_label != 'Normal':
                final_prediction_label = xgb_prediction_label
                final_confidence = xgb_confidence
            elif dl_prediction_label != 'Normal':
                final_prediction_label = dl_prediction_label
                final_confidence = dl_confidence
            else:
                final_prediction_label = 'Normal'
                final_confidence = max(xgb_confidence, dl_confidence)

            current_time_iso = datetime.datetime.now().isoformat()
            
            # --- Logique de sévérité ---
            if final_prediction_label == 'Normal':
                severity = 'normal'
            elif final_confidence > 0.95:
                severity = 'critical'
            elif final_confidence > 0.75:
                severity = 'elevated'
            else:
                severity = 'warning'

            # Compter les anomalies
            if final_prediction_label != 'Normal':
                anomaly_count += 1

            # --- FORMAT COMPATIBLE AVEC LE SERVEUR NODE.JS ---
            alert_payload = {
                "machineId": MACHINE_ID,  # Utiliser l'ID configuré
                "type": final_prediction_label,
                "severity": severity,
                "message": f"Anomalie '{final_prediction_label}' (Confiance: {final_confidence:.2f}) détectée sur {MACHINE_ID}.",
                "timestamp": current_time_iso,
                "details": {
                    "xgb_prediction": xgb_prediction_label,
                    "xgb_confidence": float(xgb_confidence),
                    "dl_prediction": dl_prediction_label,
                    "dl_confidence": float(dl_confidence),
                    "raw_data_sample": current_window_df.iloc[-1].to_dict()
                }
            }
            
            # Essayer MQTT d'abord
            sent_via_mqtt = False
            try:
                if mqtt_client:
                    client.publish(MQTT_TOPIC_PREDICTION_OUT, json.dumps(alert_payload))
                    sent_via_mqtt = True
            except Exception as e:
                print(f"⚠️ Échec MQTT pour prédiction: {e}")
            
            # Fallback HTTP si MQTT échoue
            if not sent_via_mqtt:
                send_http_fallback("api/raspberry-pi/prediction", alert_payload)
            
            print(f"🔴 [{current_time_iso}] Prédiction: {final_prediction_label} (Conf: {final_confidence:.2f}, Sévérité: {severity})")
            print(f"   📊 XGBoost: {xgb_prediction_label} ({xgb_confidence:.2f}) | CNN-LSTM: {dl_prediction_label} ({dl_confidence:.2f})")
            if final_prediction_label != 'Normal':
                print("   🚨 ALERTE ENVOYÉE AU SERVEUR!")

    except json.JSONDecodeError:
        print(f"❌ [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Erreur JSON: {msg.payload.decode('utf-8')}")
    except Exception as e:
        print(f"❌ [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Erreur: {e}")

# --- Tâches périodiques ---
def periodic_tasks():
    """Tâches exécutées périodiquement en arrière-plan"""
    global last_metrics_time
    
    while True:
        try:
            current_time = time.time()
            
            # Envoyer les métriques
            if current_time - last_metrics_time >= METRICS_INTERVAL_SECONDS:
                send_metrics_update()
                last_metrics_time = current_time
            
            # Envoyer le statut
            if current_time % STATUS_INTERVAL_SECONDS == 0:
                send_device_status()
                
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Erreur dans les tâches périodiques: {e}")
            time.sleep(5)

# --- Gestion de l'arrêt gracieux ---
def signal_handler(sig, frame):
    print(f"\n🛑 Arrêt du système détecté...")
    
    # Envoyer un statut offline
    try:
        if mqtt_client:
            offline_status = {
                "device_id": MACHINE_ID,
                "status": "offline", 
                "timestamp": datetime.datetime.now().isoformat()
            }
            mqtt_client.publish(MQTT_TOPIC_DEVICE_STATUS, json.dumps(offline_status))
            mqtt_client.disconnect()
    except:
        pass
    
    print("✅ Arrêt complet.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Fonction principale ---
def main():
    global start_time
    start_time = time.time()
    
    print("🚀 Démarrage du système de prédiction Raspberry Pi...")
    print(f"🔧 Machine ID: {MACHINE_ID}")
    print(f"📡 Topic d'entrée: {MQTT_TOPIC_DATA_IN}")
    print(f"📤 Topics de sortie:")
    print(f"   - Capteur: {MQTT_TOPIC_SENSOR_OUT}")
    print(f"   - Prédiction: {MQTT_TOPIC_PREDICTION_OUT}")
    print(f"   - Métriques: {MQTT_TOPIC_METRICS_OUT}")
    print(f"📊 Taille de fenêtre: {WINDOW_SIZE_SECONDS}s (~{EXPECTED_SAMPLES_PER_WINDOW} échantillons)")
    print(f"🔄 HTTP Fallback: {'Activé' if HTTP_FALLBACK_ENABLED else 'Désactivé'}")
    print("⌨️  Appuyez sur Ctrl+C pour arrêter.\n")

    # Démarrer les tâches périodiques en arrière-plan
    periodic_thread = threading.Thread(target=periodic_tasks, daemon=True)
    periodic_thread.start()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"❌ Erreur MQTT: {e}")
        if HTTP_FALLBACK_ENABLED:
            print("🔄 Basculement en mode HTTP uniquement...")
            # Ici vous pourriez implémenter une boucle HTTP pure si nécessaire
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()
