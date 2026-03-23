#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Servo.h>

// ===== WIFI =====
const char* ssid = "Nothing Phone3";
const char* password = "Nothing1";

ESP8266WebServer server(80);

// ===== PINS =====
#define TRIG D1
#define ECHO D2
#define SERVO_PIN D3

Servo myServo;

// ===== SETTINGS =====
#define DETECT_MIN 2
#define DETECT_MAX 20

unsigned long lastReadTime = 0;
unsigned long READ_INTERVAL = 10000;

bool objectDetected = false;

// ===== NEW STATUS VARIABLES =====
String lastObjectType = "NONE";
String lastAction = "IDLE";

// ===== DISTANCE =====
int getDistance() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long duration = pulseIn(ECHO, HIGH, 30000);
  int dist = duration * 0.034 / 2;
  if (dist == 0 || dist > 400) return 999;
  return dist;
}

// ===== SENSOR LOGIC =====
void updateDetection() {
  int distance = getDistance();
  Serial.print("Distance: ");
  Serial.println(distance);
  if (distance >= DETECT_MIN && distance <= DETECT_MAX) {
    objectDetected = true;
    Serial.println("📦 OBJECT DETECTED");
  } else {
    objectDetected = false;
  }
}

// ===== CHECK =====
void handleCheck() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  if (objectDetected) server.send(200, "text/plain", "DETECTED");
  else server.send(200, "text/plain", "NONE");
}

// ===== ACTIONS =====
void handleBio() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  myServo.write(30);
  lastObjectType = "BIOLOGICAL";
  lastAction = "Servo moved LEFT";
  Serial.println("♻️ BIOLOGICAL RECEIVED");
  Serial.println("↩️ Servo LEFT");
  server.send(200, "text/plain", "ACTION_DONE:Biological");
}

void handleDry() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  myServo.write(90);
  lastObjectType = "DRY";
  lastAction = "Servo CENTER";
  Serial.println("📦 DRY RECEIVED");
  Serial.println("↩️ Servo CENTER");
  server.send(200, "text/plain", "ACTION_DONE:Dry");
}

void handleMetal() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  myServo.write(150);
  lastObjectType = "METAL";
  lastAction = "Servo RIGHT";
  Serial.println("🔩 METAL RECEIVED");
  Serial.println("↩️ Servo RIGHT");
  server.send(200, "text/plain", "ACTION_DONE:Metal");
}

// ===== STATUS API =====
void handleStatus() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  String msg = "Object: " + lastObjectType + "\n" + "Action: " + lastAction;
  server.send(200, "text/plain", msg);
}

// ===== ROOT =====
void handleRoot() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "text/plain", "ESP READY");
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  myServo.attach(SERVO_PIN);
  myServo.write(90);
  WiFi.begin(ssid, password);
  Serial.print("Connecting...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  Serial.println("\n✅ WiFi Connected");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/check", handleCheck);
  server.on("/status", handleStatus);
  server.on("/BIOLOGICAL", handleBio);
  server.on("/DRY", handleDry);
  server.on("/METAL", handleMetal);

  server.begin();
}

// ===== LOOP =====
void loop() {
  if (millis() - lastReadTime > READ_INTERVAL) {
    updateDetection();
    lastReadTime = millis();
  }
  server.handleClient();
}