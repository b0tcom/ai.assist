#include <Mouse.h>

String serialBuffer = "";

void setup() {
  Serial.begin(115200);
  Mouse.begin();
  serialBuffer.reserve(32);  // Reserve space for command strings
}

void loop() {
  // Read from serial and accumulate until newline
  while (Serial.available() > 0) {
    char receivedChar = Serial.read();
    if (receivedChar == '\n') {
      processCommand(serialBuffer);
      serialBuffer = "";  // Reset buffer
    } else {
      serialBuffer += receivedChar;
      // Avoid overflow if no newline is sent
      if (serialBuffer.length() > 64) serialBuffer = "";
    }
  }
}

void processCommand(const String &command) {
  if (command.length() == 0) return;

  // Debug: Echo received command
  Serial.print("Received: ");
  Serial.println(command);

  // Mouse move: "m,dx,dy"
  if (command.startsWith("m,")) {
    int firstComma = command.indexOf(',');
    int secondComma = command.indexOf(',', firstComma + 1);
    if (firstComma > 0 && secondComma > firstComma) {
      int dx = command.substring(firstComma + 1, secondComma).toInt();
      int dy = command.substring(secondComma + 1).toInt();
      Mouse.move(dx, dy, 0);
      // Debug: Echo action
      Serial.print("Moved mouse by: ");
      Serial.print(dx);
      Serial.print(", ");
      Serial.println(dy);
    }
  }
  // Recoil: "r,strength"
  else if (command.startsWith("r,")) {
    int firstComma = command.indexOf(',');
    if (firstComma > 0) {
      int strength = command.substring(firstComma + 1).toInt();
      Mouse.move(0, strength, 0);
      // Debug: Echo action
      Serial.print("Applied recoil: ");
      Serial.println(strength);
    }
  }
  // Add additional commands here, if needed.
  // else if (command.startsWith("jitter,")) { ... }
}
