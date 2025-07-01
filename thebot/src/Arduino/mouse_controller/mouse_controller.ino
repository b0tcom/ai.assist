/*
 * CV Targeting System - Arduino Mouse Controller
 * ---------------------------------------------
 * This sketch listens for serial commands from the Python host application
 * and translates them into mouse movements.
 *
 * It requires an Arduino with native USB support (ATmega32U4 based),
 * such as the Arduino Leonardo or Pro Micro.
 *
 * Serial Command Protocol:
 * - Move: "m,dx,dy\n" (e.g., "m,10,-5\n")
 * - Recoil: "r,strength\n" (e.g., "r,5\n")
 */

#include <Mouse.h>
#include <HID.h>

String serialBuffer = "";

void setup() {
  // Start serial communication at the baud rate specified in the Python config.
  Serial.begin(115200);
  // Initialize the mouse library.
  Mouse.begin();
  serialBuffer.reserve(32);  // Pre-allocate buffer memory
}

void loop() {
  // Read from serial port if data is available.
  while (Serial.available() > 0) {
    char receivedChar = Serial.read();

    // Process the command when a newline character is received.
    if (receivedChar == '\n') {
      processCommand(serialBuffer);
      serialBuffer = "";  // Clear the buffer for the next command
    } else {
      serialBuffer += receivedChar;  // Append character to the buffer
    }
  }
}

void processCommand(String command) {
  // Check the command type (e.g., 'm' for move, 'r' for recoil).
  if (command.startsWith("m,")) {
    parseAndMove(command);
  } else if (command.startsWith("r,")) {
    applyRecoil(command);
  }
}

void parseAndMove(String command) {
  // Find the positions of the delimiters.
  int firstComma = command.indexOf(',');
  int secondComma = command.indexOf(',', firstComma + 1);

  // Extract the dx and dy values.
  String dxStr = command.substring(firstComma + 1, secondComma);
  String dyStr = command.substring(secondComma + 1);

  // Convert to integers and move the mouse.
  int dx = dxStr.toInt();
  int dy = dyStr.toInt();
  Mouse.move(dx, dy, 0);
}

void applyRecoil(String command) {
  // Find the position of the delimiter.
  int firstComma = command.indexOf(',');

  // Extract the recoil strength value.
  String strengthStr = command.substring(firstComma + 1);
  int strength = strengthStr.toInt();

  // Apply recoil by moving the mouse down.
  Mouse.move(0, strength, 0);
}