#include "df4MotorDriver.h"


void StartPumpA(uint16_t speed, int dir){
    //Setting initial velocity(Min:0  Max:127)
    uint16_t speedCorrected = map(speed,0,4096,0,4096);
    digitalWrite(DFDRIVER_DIR_1,dir);
    analogWrite(DFDRIVER_PWM_1,speedCorrected);
}

void StartPumpB(uint16_t speed, int dir){
    //Setting initial velocity(Min:0  Max:127)
    uint16_t speedCorrected = map(speed,0,4096,0,4096);
    digitalWrite(DFDRIVER_DIR_2,dir);
    analogWrite(DFDRIVER_PWM_2,speedCorrected);
}

void StartPumpC(uint16_t speed, int dir){
    //Setting initial velocity(Min:0  Max:127)
    uint16_t speedCorrected = map(speed,0,4096,0,4096);
    digitalWrite(DFDRIVER_DIR_3,dir);
    analogWrite(DFDRIVER_PWM_3,speedCorrected);
}


void StopPumpA(){
   analogWrite(DFDRIVER_PWM_1,0);   
}

void StopPumpB(){
   analogWrite(DFDRIVER_PWM_2,0);   
}

void StopPumpC(){
   analogWrite(DFDRIVER_PWM_3,0);   
}


void setupPumps(){
    analogWriteResolution(12);
    pinMode(DFDRIVER_PWM_1,OUTPUT);
    pinMode(DFDRIVER_PWM_2,OUTPUT);
    pinMode(DFDRIVER_PWM_3,OUTPUT);
    pinMode(DFDRIVER_PWM_4,OUTPUT);
    pinMode(DFDRIVER_DIR_1,OUTPUT);
    pinMode(DFDRIVER_DIR_2,OUTPUT);
    pinMode(DFDRIVER_DIR_3,OUTPUT);
    pinMode(DFDRIVER_DIR_4,OUTPUT);
}

void stopPumps(){
  pumpA.state = false;
  StopPumpA();
  pumpB.state = false;
  StopPumpB();
  pumpC.state = false;
  StopPumpC();
}

Step currentStep;

