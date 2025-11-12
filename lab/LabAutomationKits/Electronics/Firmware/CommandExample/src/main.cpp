#include <Arduino.h>
#include <CmdMessenger.h>
#include "dataModel.h"
#include "df4MotorDriver.h"

CmdMessenger cmdMessenger(Serial, ',', ';', '/');

// This is the list of recognized commands. These can be commands that can either be sent or received.
// In order to receive, attach a callback function to these events
void OnUnknownCommand();
void OnWatchdogRequest();
void OnArduinoReady();
void OnReceiveStart();
void OnReceiveStop();

void receiveStart();
void receiveStop();

enum
{
  // Commands
  kWatchdog,          // Command to request application ID
  kAcknowledge,       // Command to acknowledge a received command
  kError,             // Command to message that an error has occurred
  kStart,              // Command to receive a step (pump state + time), should always contain the full state of the pumps
  kStop,              // Command to stop all pumps

};

// Commands we send from the PC and want to receive on the Arduino.
// We must define a callback function in our Arduino program for each entry in the list below.
void attachCommandCallbacks()
{
  // Attach callback methods
  cmdMessenger.attach(OnUnknownCommand);
  cmdMessenger.attach(kWatchdog, OnWatchdogRequest);
  cmdMessenger.attach(kStart, OnReceiveStart);
  cmdMessenger.attach(kStop, OnReceiveStop);
}

// ------------------  C A L L B A C K S -----------------------

// Called when a received command has no attached function
void OnUnknownCommand()
{
  cmdMessenger.sendCmd(kError, "Command without attached callback");
}

void OnWatchdogRequest()
{
  // Will respond with same command ID and Unique device identifier.
  cmdMessenger.sendCmd(kWatchdog, "0000000-0000-0000-0000-00000000001");
}

// Callback function that responds that Arduino is ready (has booted up)
void OnArduinoReady()
{
  cmdMessenger.sendCmd(kAcknowledge, "Arduino ready");
}

void OnReceiveStart()
{
  receiveStart();
}

void OnReceiveStop()
{
  receiveStop();
}



void setup() {
  Serial.begin(115200);
  setupPumps();

  //  Do not print newLine at end of command,
  //  in order to reduce data being sent
  cmdMessenger.printLfCr(false);

  // Attach my application's user-defined callback methods
  attachCommandCallbacks();

  cmdMessenger.sendCmd(kAcknowledge, "Arduino has started!");
}

void loop() {

  cmdMessenger.feedinSerialData();

}


void receiveStart()
{
    pumpA.state = cmdMessenger.readBinArg<bool>();
    pumpA.speed = cmdMessenger.readBinArg<uint16_t>();
    pumpA.dir = cmdMessenger.readBinArg<bool>();

    if (pumpA.state)
    {
      StartPumpA(pumpA.speed, pumpA.dir);
    }
    else
    {
      StopPumpA();
    }

}

void receiveStop()
{

  stopPumps();
  
  cmdMessenger.sendCmd(kAcknowledge, "Stopped");
}