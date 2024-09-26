#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>

// Wi-Fi credentials
const char* ssid = "Amshu GT 2";
const char* password = "Amshu@399";

// WebSocket server on port 81
WebSocketsServer webSocket = WebSocketsServer(81);

// Motor A pins
#define IN1 16
#define IN2 17
#define ENA 21  // Motor A enable pin (PWM for speed control)

// Motor B pins
#define IN3 18
#define IN4 19
#define ENB 22  // Motor B enable pin (PWM for speed control)

// Define distance per matrix cell in cm
#define CELL_DISTANCE 5

// WebSocket event handler
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  if (type == WStype_TEXT) {
    
    

    if (String((char*)payload) == "turnLeft") {
      Serial.print("Received: ");
      Serial.println(String((char*)payload));
      turnLeft();
    }
    else if (String((char*)payload) == "turnRight") {
      Serial.print("Received: ");
      Serial.println(String((char*)payload));
      turnRight();
    }
    else if (String((char*)payload) == "moveForward") {
      Serial.print("Received: ");
      Serial.println(String((char*)payload));
      moveForward(CELL_DISTANCE);
    }
  }
}

// Function to move forward a given distance
void moveForward(int distance) {
  int time = calculateTime(distance);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  delay(time);
  stopMotors();
}

// Function to turn left
void turnLeft() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  delay(500);  // Adjust for a proper 90-degree turn
  stopMotors();
}

// Function to turn right
void turnRight() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  delay(500);  // Adjust for a proper 90-degree turn
  stopMotors();
}

// Stop all motors
void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

// Function to calculate the time needed to move forward based on distance
int calculateTime(int distance) {
  int time = (distance / 5) * 1000;  // Assume 1 second for 10 cm
  return time;
}

// Setup Wi-Fi connection
void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);

  // Set motor pins as outputs
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Set motor speed (can use PWM for speed control, e.g., 255 = full speed)
  analogWrite(ENA, 255);  // Full speed for motor A
  analogWrite(ENB, 255);  // Full speed for motor B

  // Setup Wi-Fi
  setup_wifi();

  // Start WebSocket server
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);  // Register event handler
}

void loop() {
  // Handle WebSocket communication
  webSocket.loop();
}
