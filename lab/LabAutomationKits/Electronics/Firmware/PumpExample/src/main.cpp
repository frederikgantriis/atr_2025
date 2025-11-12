#include <Arduino.h>
#include "dataModel.h"
#include "df4MotorDriver.h"

void setup() {
  Serial.begin(115200);
  Serial.println("Setup started");
  setupPumps();
  Serial.println("Setup finished");
}

void loop() {
  delay(3000);
  Serial.println("Starting pump A forward at full speed for 3 seconds");
  StartPumpA(4096, HIGH);

  delay(3000);
  Serial.println("Starting pump A forward at half speed for 3 seconds");
  StartPumpA(2048, HIGH);

  delay(3000);
  Serial.println("Starting pump B forward at half speed for 3 seconds");
  StartPumpB(2048, HIGH);

  delay(3000);
  Serial.println("Starting pump C forward at half speed for 3 seconds");
  StartPumpC(2048, HIGH);

  delay(3000);
  stopPumps();
  Serial.println("Pumps stopped");
}