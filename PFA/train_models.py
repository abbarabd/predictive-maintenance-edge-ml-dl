import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D, Flatten

import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
LABELED_DATA_FILE = 'sensor_data_labeled.csv' # Fichier CSV labellisé
FEATURES_XGB_FILE = 'features_for_xgboost.csv' # Fichier de caractéristiques pour XGBoost

# Taille de la fenêtre en secondes (DOIT CORRESPONDRE À L'ESP32 ET AU DÉPLOIEMENT)
WINDOW_SIZE_SECONDS = 1
# Intervalle de lecture de l'ESP32 en ms (DOIT CORRESPONDRE À VOTRE CODE ESP32)
ESP32_READ_INTERVAL_MS = 20
# Nombre d'échantillons attendus par fenêtre
EXPECTED_SAMPLES_PER_WINDOW = int(WINDOW_SIZE_SECONDS * (1000 / ESP32_READ_INTERVAL_MS))

# Colonnes de données brutes à utiliser pour le CNN-LSTM (doit correspondre à dl_raw_data_columns.pkl)
RAW_DATA_COLS = ['temperature_c', 'accel_x_g', 'accel_y_g', 'accel_z_g', 'raw_sound_analog']


# --- Charger les données labellisées pour récupérer les labels et l'encodeur ---
df_raw_for_labels = pd.read_csv(LABELED_DATA_FILE)
label_encoder = LabelEncoder()
label_encoder.fit(df_raw_for_labels['fault_type'].unique()) # Fit sur tous les labels uniques

print("Correspondance des labels numériques :")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")

# --- 1. Préparation et Entraînement XGBoost ---
print("\n--- Préparation et Entraînement XGBoost ---")
features_df_xgb = pd.read_csv(FEATURES_XGB_FILE)
X_xgb = features_df_xgb.drop('label', axis=1)
y_xgb_encoded = label_encoder.transform(features_df_xgb['label'])

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb_encoded, test_size=0.2, random_state=42, stratify=y_xgb_encoded
)

print(f"Taille de l'ensemble d'entraînement (X_train_xgb) : {X_train_xgb.shape}")
print(f"Taille de l'ensemble de test (X_test_xgb) : {X_test_xgb.shape}")

model_xgb = xgb.XGBClassifier(objective='multi:softmax',
                              num_class=len(label_encoder.classes_),
                              eval_metric='mlogloss',
                              n_estimators=100,
                              learning_rate=0.1,
                              max_depth=5,
                              random_state=42)

print("Entraînement du modèle XGBoost...")
model_xgb.fit(X_train_xgb, y_train_xgb)
print("Entraînement XGBoost terminé.")

y_pred_xgb = model_xgb.predict(X_test_xgb)
print("\n--- Rapports d'évaluation XGBoost ---")
print("Accuracy:", accuracy_score(y_test_xgb, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test_xgb, y_pred_xgb, target_names=label_encoder.classes_))

cm_xgb = confusion_matrix(y_test_xgb, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Matrice de Confusion (XGBoost)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

plt.figure(figsize=(12, 6))
xgb.plot_importance(model_xgb, max_num_features=20)
plt.title('Importance des Caractéristiques (XGBoost)')
plt.show()

# --- 2. Préparation et Entraînement CNN-LSTM ---
print("\n--- Préparation et Entraînement CNN-LSTM ---")

# Recharger les données brutes pour le CNN-LSTM (pour s'assurer de la cohérence)
df_dl_full = pd.read_csv(LABELED_DATA_FILE)
df_dl_full['timestamp_rpi'] = pd.to_datetime(df_dl_full['timestamp_rpi'])
df_dl_full = df_dl_full.set_index('timestamp_rpi')
df_dl_full = df_dl_full.fillna(method='ffill').fillna(method='bfill').fillna(0)

X_dl = []
y_dl = []

for i in range(0, len(df_dl_full) - EXPECTED_SAMPLES_PER_WINDOW + 1, EXPECTED_SAMPLES_PER_WINDOW):
    window_data = df_dl_full.iloc[i : i + EXPECTED_SAMPLES_PER_WINDOW]
    if len(window_data) == EXPECTED_SAMPLES_PER_WINDOW:
        raw_features = window_data[RAW_DATA_COLS].values
        X_dl.append(raw_features)
        y_dl.append(window_data['fault_type'].mode()[0])

X_dl = np.array(X_dl)
y_dl = np.array(y_dl)
y_dl_encoded = label_encoder.transform(y_dl)

scaler_dl = StandardScaler()
X_dl_reshaped = X_dl.reshape(-1, X_dl.shape[-1])
X_dl_scaled = scaler_dl.fit_transform(X_dl_reshaped)
X_dl_scaled = X_dl_scaled.reshape(X_dl.shape)

X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
    X_dl_scaled, y_dl_encoded, test_size=0.2, random_state=42, stratify=y_dl_encoded
)

print(f"Taille de l'ensemble d'entraînement (X_train_dl) : {X_train_dl.shape}")
print(f"Taille de l'ensemble de test (X_test_dl) : {X_test_dl.shape}")

model_dl = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_dl.shape[1], X_train_dl.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    LSTM(100, return_sequences=False),
    Dropout(0.3),

    Dense(len(label_encoder.classes_), activation='softmax')
])

model_dl.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Entraînement du modèle CNN-LSTM...")
history_dl = model_dl.fit(X_train_dl, y_train_dl, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
print("Entraînement CNN-LSTM terminé.")

loss_dl, accuracy_dl = model_dl.evaluate(X_test_dl, y_test_dl, verbose=0)
print(f"\n--- Rapports d'évaluation CNN-LSTM ---")
print(f"Test Accuracy: {accuracy_dl:.4f}")

y_pred_dl = np.argmax(model_dl.predict(X_test_dl), axis=-1)
print("\nClassification Report (CNN-LSTM):\n", classification_report(y_test_dl, y_pred_dl, target_names=label_encoder.classes_))

cm_dl = confusion_matrix(y_test_dl, y_pred_dl)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dl, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Matrice de Confusion (CNN-LSTM)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

# --- 3. Sauvegarde des Modèles et des Outils ---
joblib.dump(model_xgb, 'xgboost_motor_model.pkl')
model_dl.save('cnn_lstm_motor_model.h5')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(X_xgb.columns.tolist(), 'xgb_feature_columns.pkl')
joblib.dump(RAW_DATA_COLS, 'dl_raw_data_columns.pkl')
joblib.dump(scaler_dl, 'dl_scaler.pkl')

print("\nModèles et outils sauvegardés.")
