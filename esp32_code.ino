// NeuroType ESP32 + ADS1115

#include <WiFi.h>
#include <WebSocketsServer.h>
#include <Adafruit_ADS1X15.h>
#include <ArduinoJson.h>

const char* ssid = "Nhung";
const char* password = "Nhung-2611";

WebSocketsServer webSocket(81);

Adafruit_ADS1115 ads;

unsigned long lastSampleTime = 0;
const uint32_t sampleIntervalMs = 2;

void webSocketEvent(uint8_t num,
                    WStype_t type,
                    uint8_t * payload,
                    size_t length) {

  switch (type) {

    case WStype_DISCONNECTED:
      Serial.printf("[%u] Client disconnected\n", num);
      break;

    case WStype_CONNECTED: {
      IPAddress ip = webSocket.remoteIP(num);

      Serial.printf(
        "[%u] Client connected from %d.%d.%d.%d\n",
        num,
        ip[0],
        ip[1],
        ip[2],
        ip[3]
      );
      break;
    }

    case WStype_TEXT:
      Serial.printf("[%u] Received: %s\n", num, payload);
      break;

    default:
      break;
  }
}

void setup() {

  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("================================");
  Serial.println(" NeuroType ESP32 Starting...");
  Serial.println("================================");

  WiFi.begin(ssid, password);

  Serial.print("Connecting WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi Connected!");

  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  webSocket.begin();
  webSocket.onEvent(webSocketEvent);

  Serial.println("WebSocket Server Started");
  Serial.println("Port: 81");

  if (!ads.begin()) {

    Serial.println("ADS1115 NOT FOUND!");

    while (1) {
      delay(1000);
    }
  }

  ads.setGain(GAIN_TWO);

  ads.setDataRate(RATE_ADS1115_475SPS);

  Serial.println("ADS1115 READY");
}

void loop() {

  webSocket.loop();

  unsigned long now = millis();

  if (now - lastSampleTime >= sampleIntervalMs) {

    lastSampleTime = now;

    int16_t eeg_raw =
      ads.readADC_SingleEnded(0);

    int16_t eog_raw =
      ads.readADC_SingleEnded(1);

    JsonDocument doc;

    doc["t"] = now;
    doc["e"] = eog_raw;
    doc["b"] = eeg_raw;

    String jsonString;

    serializeJson(doc, jsonString);

    webSocket.broadcastTXT(jsonString);
  }
}

