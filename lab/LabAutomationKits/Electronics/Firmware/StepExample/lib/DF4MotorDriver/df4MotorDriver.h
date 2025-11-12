#ifndef DF4MOTORDRIVER_H
#define DF4MOTORDRIVER_H
#include <Arduino.h>
#include "dataModel.h"


#define DFDRIVER_PWM_1 8
#define DFDRIVER_PWM_2 16
#define DFDRIVER_PWM_3 10
#define DFDRIVER_PWM_4 11

#define DFDRIVER_DIR_1 9
#define DFDRIVER_DIR_2 21
#define DFDRIVER_DIR_3 13
#define DFDRIVER_DIR_4 12

class Step {
    public:
        bool stateA;
        uint16_t speedA;
        bool dirA;
        bool stateB;
        uint16_t speedB;
        bool dirB;
        bool stateC;
        uint16_t speedC;
        bool dirC;
        unsigned long time;
        bool state = false;
        bool done = false;
        unsigned long stepStartTime;
};

extern Step currentStep;

void StartPumpA(uint16_t speed, int dir);
void StartPumpB(uint16_t speed, int dir);
void StartPumpC(uint16_t speed, int dir);


void StopPumpA();
void StopPumpB();
void StopPumpC();

void setupPumps();
void stopPumps();


#endif