import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq # Pour la FFT si vous allez plus loin
import joblib # Pour sauvegarder les scalers, encodeurs, etc.

# --- Configuration ---
RAW_DATA_FILE = 'sensor_data_collection.csv' # Fichier CSV brut du RPi
LABELED_DATA_FILE = 'sensor_data_labeled.csv' # Fichier CSV de sortie avec labels
FEATURES_XGB_FILE = 'features_for_xgboost.csv' # Fichier de caractéristiques pour XGBoost

# Taille de la fenêtre en secondes (DOIT CORRESPONDRE À L'ESP32 ET AU DÉPLOIEMENT)
WINDOW_SIZE_SECONDS = 1
# Intervalle de lecture de l'ESP32 en ms (DOIT CORRESPONDRE À VOTRE CODE ESP32)
ESP32_READ_INTERVAL_MS = 100
# Nombre d'échantillons attendus par fenêtre
EXPECTED_SAMPLES_PER_WINDOW = int(WINDOW_SIZE_SECONDS * (1000 / ESP32_READ_INTERVAL_MS))

# Colonnes de données brutes à utiliser pour le CNN-LSTM
# Notez que nous utilisons l'amplitude du son et non une valeur brute
RAW_DATA_COLS = ['temperature_c', 'accel_x_g', 'accel_y_g', 'accel_z_g', 'sound_rms']


# --- 1. Charger et Pré-traiter les Données Brutes ---
print("Chargement des données brutes...")
df = pd.read_csv(RAW_DATA_FILE)

# Convertir la colonne de timestamp en format datetime et la définir comme index
df['timestamp_rpi'] = pd.to_datetime(df['timestamp_rpi'])
df = df.set_index('timestamp_rpi')

# Gérer les valeurs sentinelles (-1.0 pour température)
df['temperature_c'] = df['temperature_c'].replace(-1.0, np.nan)
df = df.fillna(method='ffill').fillna(method='bfill') # Remplir les NaN avec la dernière/première valeur
df = df.fillna(0) # Remplir les NaN restants (si au début de série) avec 0

print("Données brutes chargées et pré-traitées.")
print(df.head())

# --- 2. Labellisation Manuelle (Basée sur votre Journal de Bord) ---
# C'est la partie où vous devez éditer le code en fonction de votre collecte !
print("\nDébut de la labellisation manuelle...")
df['fault_type'] = 'Normal' # Initialiser toutes les données comme 'Normal'

# Exemple de périodes d'anomalies (À ADAPTER AVEC VOS PROPRES TIMESTAMPS ET TYPES D'ANOMALIES)
# Utilisez les timestamps de votre journal de bord !
# Format: pd.to_datetime('AAAA-MM-JJ HH:MM:SS.ms')
anomalies = [
    {'start': '2025-07-27 10:15:00.000', 'end': '2025-07-27 10:20:00.000', 'type': 'Unbalance'},
    {'start': '2025-07-27 10:25:00.000', 'end': '2025-07-27 10:30:00.000', 'type': 'Friction'},
    {'start': '2025-07-27 10:35:00.000', 'end': '2025-07-27 10:40:00.000', 'type': 'Abnormal_Noise'},
    # Ajoutez d'autres périodes d'anomalies ici
]

for anomaly in anomalies:
    start_ts = pd.to_datetime(anomaly['start'])
    end_ts = pd.to_datetime(anomaly['end'])
    df.loc[(df.index >= start_ts) & (df.index < end_ts), 'fault_type'] = anomaly['type']

print("\nLabellisation terminée. Distribution des labels :")
print(df['fault_type'].value_counts())

# Sauvegarder le fichier labellisé
df.to_csv(LABELED_DATA_FILE, index=True) # Index=True pour conserver le timestamp comme colonne
print(f"\nDonnées labellisées sauvegardées dans '{LABELED_DATA_FILE}'.")

# --- 3. Ingénierie des Caractéristiques pour XGBoost ---
print("\nDébut de l'ingénierie des caractéristiques pour XGBoost...")
features_list_xgb = []

for window_start, window_df in df.resample(f'{WINDOW_SIZE_SECONDS}S', origin='start_day', closed='right', label='right'):
    if not window_df.empty:
        window_label = window_df['fault_type'].mode()[0]

        # Assurez-vous que la fenêtre a suffisamment de données pour les calculs statistiques
        if len(window_df) < 2:
            pass

        features = {
            'label': window_label,
            'temp_mean': window_df['temperature_c'].mean(),
            'temp_std': window_df['temperature_c'].std(),
            'temp_min': window_df['temperature_c'].min(),
            'temp_max': window_df['temperature_c'].max(),

            # On utilise le RMS du son
            'sound_rms_mean': window_df['sound_rms'].mean(),
            'sound_rms_std': window_df['sound_rms'].std(),
            'sound_rms_max': window_df['sound_rms'].max(),

            'accel_x_rms': np.sqrt(np.mean(window_df['accel_x_g']**2)),
            'accel_y_rms': np.sqrt(np.mean(window_df['accel_y_g']**2)),
            'accel_z_rms': np.sqrt(np.mean(window_df['accel_z_g']**2)),
            
            'accel_x_std': window_df['accel_x_g'].std(),
            'accel_y_std': window_df['accel_y_g'].std(),
            'accel_z_std': window_df['accel_z_g'].std(),

            'accel_x_ptp': window_df['accel_x_g'].max() - window_df['accel_x_g'].min(),
            'accel_y_ptp': window_df['accel_y_g'].max() - window_df['accel_y_g'].min(),
            'accel_z_ptp': window_df['accel_z_g'].max() - window_df['accel_z_g'].min(),

            'accel_x_skew': skew(window_df['accel_x_g']),
            'accel_y_skew': skew(window_df['accel_y_g']),
            'accel_z_skew': skew(window_df['accel_z_g']),

            'accel_x_kurt': kurtosis(window_df['accel_x_g']),
            'accel_y_kurt': kurtosis(window_df['accel_y_g']),
            'accel_z_kurt': kurtosis(window_df['accel_z_g']),
        }
        features_list_xgb.append(features)

features_df_xgb = pd.DataFrame(features_list_xgb)
features_df_xgb = features_df_xgb.fillna(0.0) # Remplir les NaN résultant des calculs de stats par 0

print("\nDataFrame des caractéristiques pour XGBoost créé.")
print(features_df_xgb.head())
print(f"Nombre de fenêtres (échantillons pour XGBoost) : {len(features_df_xgb)}")

# Sauvegarder les caractéristiques pour l'entraînement
features_df_xgb.to_csv(FEATURES_XGB_FILE, index=False)
print(f"Caractéristiques pour XGBoost sauvegardées dans '{FEATURES_XGB_FILE}'.")

# Sauvegarder les colonnes de features pour le déploiement
xgb_feature_columns = features_df_xgb.drop('label', axis=1).columns.tolist()
joblib.dump(xgb_feature_columns, 'xgb_feature_columns.pkl')
print("Ordre des colonnes de caractéristiques XGBoost sauvegardé.")

# --- 4. Préparation des Données Brutes pour CNN-LSTM ---
print("\nPréparation des données brutes pour CNN-LSTM...")

X_dl = []
y_dl = []

# Itérer sur le DataFrame avec une fenêtre glissante (non-chevauchante pour l'exemple)
for i in range(0, len(df) - EXPECTED_SAMPLES_PER_WINDOW + 1, EXPECTED_SAMPLES_PER_WINDOW):
    window_data = df.iloc[i : i + EXPECTED_SAMPLES_PER_WINDOW]
    
    if len(window_data) == EXPECTED_SAMPLES_PER_WINDOW:
        raw_features = window_data[RAW_DATA_COLS].values
        X_dl.append(raw_features)
        y_dl.append(window_data['fault_type'].mode()[0]) # Label de la fenêtre

X_dl = np.array(X_dl)
y_dl = np.array(y_dl)

joblib.dump(RAW_DATA_COLS, 'dl_raw_data_columns.pkl') # Sauvegarde l'ordre des colonnes brutes pour DL
print("Ordre des colonnes de données brutes pour DL sauvegardé.")

print(f"Forme de X_dl (CNN-LSTM) : {X_dl.shape}")
print(f"Forme de y_dl (CNN-LSTM) : {y_dl.shape}")
print("Préparation des données brutes pour CNN-LSTM terminée.")

# Le reste de l'entraînement (XGBoost et CNN-LSTM) sera dans le script d'entraînement séparé.
