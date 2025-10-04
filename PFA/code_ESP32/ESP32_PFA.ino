// Mesure d'amplitude sonore (crête-à-crête) dans le code MQTT existant
// Version corrigée pour un capteur de son plus pertinent

#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>
#include <Wire.h>          // Pour I2C avec le MPU6050
#include <MPU6050_light.h>   // Bibliothèque légère pour MPU6050
#include <ArduinoJson.h>   // Pour créer des messages JSON

// --- Configuration WiFi ---
const char* ssid = "inwi Home ";
const char* password = "3923424";

// --- Configuration MQTT ---
const char* mqtt_server     = "192.168.0.108";
const int   mqtt_port       = 1883;
const char* mqtt_topic_data = "machine/moteur3/data";

WiFiClient espClient;
PubSubClient client(espClient);

// --- Capteurs ---
#define DHTPIN              4
#define DHTTYPE             DHT11
DHT         dht(DHTPIN, DHTTYPE);

#define PIN_SON_ANALOG      36         // GPIO34 pour le MAX9814
#define SAMPLE_WINDOW       50         // fenêtre d'échantillonnage en ms pour le son

MPU6050 mpu(Wire);

// --- Timing ---
unsigned long lastSensorRead = 0;
// Un intervalle de 100ms est un bon compromis pour collecter des données agrégées
const unsigned long SENSOR_READ_INTERVAL_MS = 100;

// --- Fonctions WiFi/MQTT ---
void setup_wifi() {
  delay(10);
  Serial.println(); Serial.print("Connecting to WiFi "); Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("ESP32 IP Address: "); Serial.println(WiFi.localIP());
}

void reconnect_mqtt() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "ESP32Client-" + String(random(0xffff), HEX);
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc="); Serial.print(client.state());
      Serial.println(" try again in 5s");
      delay(5000);
    }
  }
}

// --- Setup ---
void setup() {
  Serial.begin(115200);
  dht.begin();

  Wire.begin();
  byte status = mpu.begin();
  if (status != 0) {
    Serial.print("MPU6050 not detected! Status: ");
    Serial.println(status);
    while (1) delay(100);
  }
  Serial.println("MPU6050 ready!");
  delay(1000);
  mpu.calcOffsets();
  Serial.println("MPU6050 calibrated.");

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
}

// --- Loop ---
void loop() {
  if (!client.connected()) reconnect_mqtt();
  client.loop();

  unsigned long now = millis();
  if (now - lastSensorRead >= SENSOR_READ_INTERVAL_MS) {
    lastSensorRead = now;

    // --- Lecture DHT11 ---
    float temperature = dht.readTemperature();
    if (isnan(temperature)) {
      Serial.println("Failed to read temperature");
      temperature = -1;
    }

    // --- Lecture MPU6050 ---
    mpu.update();
    float accX = mpu.getAccX();
    float accY = mpu.getAccY();
    float accZ = mpu.getAccZ();

    // --- Mesure amplitude sonore crête-à-crête ---
    unsigned long startMillis = millis();
    unsigned int signalMax = 0, signalMin = 4095, sample;
    while (millis() - startMillis < SAMPLE_WINDOW) {
      sample = analogRead(PIN_SON_ANALOG);
      if (sample > signalMax) signalMax = sample;
      if (sample < signalMin) signalMin = sample;
    }
    unsigned int peakToPeak = signalMax - signalMin;

    // --- Construction JSON ---
    StaticJsonDocument<256> doc;
    doc["timestamp_ms"]     = now;
    doc["temperature_c"]    = temperature;
    doc["accel_x_g"]        = accX;
    doc["accel_y_g"]        = accY;
    doc["accel_z_g"]        = accZ;
    doc["sound_amplitude"] = peakToPeak; // On utilise la mesure d'amplitude

    char payload[256];
    serializeJson(doc, payload);

    // --- Publication ---
    if (client.publish(mqtt_topic_data, payload)) {
      Serial.print("MQTT published: ");
      Serial.println(payload);
    } else {
      Serial.print("MQTT publish failed, state: ");
      Serial.println(client.state());
    }
  }
}
