#include <Arduino.h>
#include <CmdMessenger.h>
#include "dataModel.h"
#include "df4MotorDriver.h"
#include <SparkFun_AS7343.h>

SfeAS7343ArdI2C mySensor;

uint16_t myData[ksfAS7343NumChannels];


CmdMessenger cmdMessenger(Serial, ',', ';', '/');

// This is the list of recognized commands. These can be commands that can either be sent or received.
// In order to receive, attach a callback function to these events
void OnUnknownCommand();
void OnWatchdogRequest();
void OnArduinoReady();
void OnGetState();
void OnGetLastStep();
void OnReceiveStep();
void OnReceiveStop();

void returnState();
void returnLastStep();
void receiveStep();
void receiveStop();

enum
{
  // Commands
  kWatchdog,          // Command to request application ID
  kAcknowledge,       // Command to acknowledge a received command
  kError,             // Command to message that an error has occurred
  kGetState,          // Command to get the pump states
  kGetStateResult,    // Command to send the full state of the pumps
  kGetLastStep,       // Command to get the current step
  kGetLastStepResult, // Command to send the current step
  kStep,              // Command to receive a step (pump state + time), should always contain the full state of the pumps
  kStop,              // Command to stop all pumps
  kStepDone,          // Command to signal a step done

};

// Commands we send from the PC and want to receive on the Arduino.
// We must define a callback function in our Arduino program for each entry in the list below.
void attachCommandCallbacks()
{
  // Attach callback methods
  cmdMessenger.attach(OnUnknownCommand);
  cmdMessenger.attach(kWatchdog, OnWatchdogRequest);
  cmdMessenger.attach(kGetState, OnGetState);
  cmdMessenger.attach(kGetLastStep, OnGetLastStep);
  cmdMessenger.attach(kStep, OnReceiveStep);
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

void OnGetState()
{
  returnState();
}

void OnGetLastStep()
{
  returnLastStep();
}

void OnReceiveStep()
{
  receiveStep();
}

void OnReceiveStop()
{
  receiveStop();
}

void setup() {

  Serial.begin(115200);
  setupPumps();

  Wire.begin();

  mySensor.begin();

  //  Do not print newLine at end of command,
  //  in order to reduce data being sent
  cmdMessenger.printLfCr(false);

  // Attach my application's user-defined callback methods
  attachCommandCallbacks();

  cmdMessenger.sendCmd(kAcknowledge, "Arduino has started!");
 
}

void loop() {

  cmdMessenger.feedinSerialData();
  if (currentStep.state)
  {
    // Serial.println("Entering queue");
    if (!currentStep.done)
    {
      int channelsRead = mySensor.getData(myData);

      // Print the data from all the channels that were read
      for (int channel = 0; channel < channelsRead; channel++)
      {
          Serial.print(myData[channel]);
          Serial.print(",");
      }
      Serial.println();

      // Serial.println("Step not done");
      if (millis() - currentStep.stepStartTime >= currentStep.time)
      {
        currentStep.done = true;
        stopPumps();
        cmdMessenger.sendCmd(kStepDone);
      }
    }
    else
    {
      currentStep.state = false;
      // Serial.println("Deactivating queue and clearing after done");
    }
  }

}

void returnState()
{
  cmdMessenger.sendCmdStart(kGetStateResult);

  cmdMessenger.sendCmdBinArg<bool>(pumpA.state);
  cmdMessenger.sendCmdBinArg<uint16_t>(pumpA.speed);
  cmdMessenger.sendCmdBinArg<bool>(pumpA.dir);

  cmdMessenger.sendCmdEnd();
}

void returnLastStep()
{

  cmdMessenger.sendCmdStart(kGetLastStepResult);

  cmdMessenger.sendCmdBinArg<bool>(currentStep.state);
  cmdMessenger.sendCmdBinArg<bool>(currentStep.done);
  cmdMessenger.sendCmdBinArg<unsigned long>(currentStep.time);

  cmdMessenger.sendCmdBinArg<bool>(currentStep.stateA);
  cmdMessenger.sendCmdBinArg<uint16_t>(currentStep.speedA);
  cmdMessenger.sendCmdBinArg<bool>(currentStep.dirA);

  cmdMessenger.sendCmdEnd();
}

void receiveStep()
{

  if (currentStep.state && !currentStep.done)
  {
    cmdMessenger.readBinArg<bool>();
    cmdMessenger.readBinArg<uint16_t>();
    cmdMessenger.readBinArg<bool>();

    cmdMessenger.readBinArg<unsigned long>();

    cmdMessenger.sendCmd(kError, "Busy");
    // cmdMessenger.sendCmdBinArg<unsigned long>(currentStep.time);
  }
  else
  {
    currentStep.stateA = cmdMessenger.readBinArg<bool>();
    currentStep.speedA = cmdMessenger.readBinArg<uint16_t>();
    currentStep.dirA = cmdMessenger.readBinArg<bool>();

    if (currentStep.stateA)
    {
      StartPumpA(currentStep.speedA, currentStep.dirA);
    }
    else
    {
      StopPumpA();
    }

    pumpA.state = currentStep.stateA;
    pumpA.speed = currentStep.speedA;
    pumpA.dir = currentStep.dirA;

    currentStep.time = cmdMessenger.readBinArg<unsigned long>();

    currentStep.state = true;
    currentStep.done = false;
    currentStep.stepStartTime = millis();

    cmdMessenger.sendCmdStart(kAcknowledge);
    cmdMessenger.sendCmdArg("Step");
    // cmdMessenger.sendCmdBinArg<unsigned long>(currentStep.time);
    cmdMessenger.sendCmdEnd();
  }
}

void receiveStop()
{

  stopPumps();
  
  currentStep.done = true;
  currentStep.state = false;
  cmdMessenger.sendCmd(kAcknowledge, "Stopped");
}

