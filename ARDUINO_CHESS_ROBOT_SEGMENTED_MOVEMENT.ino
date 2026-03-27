#include <LobotServoController.h>

LobotServoController Marm(Serial1);
LobotServo Mservos[6];
LobotServoController Larm(Serial2);
LobotServo Lservos[6];
LobotServoController Rarm(Serial3);
LobotServo Rservos[6];

int positions[7] = {0, 0, 0, 0, 0, 0, 0};

int Lpositions[6] = {0, 0, 0, 0, 0, 0};
int Mpositions[6] = {0, 0, 0, 0, 0, 0};
int Rpositions[6] = {0, 0, 0, 0, 0, 0};

int LR_Default_Positions[6] = {1000, 495, 0, 274, 850, 511};
int M_Default_Positions[6] = {1000, 493, 862, 150, 140, 491};

long time = 0;
long new_time = 0;


void setup() {
  Serial3.begin(9600);
  Serial2.begin(9600);
  Serial1.begin(9600);
  Serial.begin(9600);

  for (int i = 0; i < 6; i++) {
    Lservos[i].ID = i + 1;
    Rservos[i].ID = i + 1; 
    Mservos[i].ID = i + 1;
  }

  time = millis();
  while (Serial.available() < 1) {
    new_time = millis();
  } 
  Serial.print("Time: ");
  Serial.println(new_time - time);

  for (int i = 0; i < 6; i ++) {
    Lservos[i].Position = LR_Default_Positions[i];
    Rservos[i].Position = LR_Default_Positions[i];
    Mservos[i].Position = M_Default_Positions[i];
  
    Lpositions[i] = LR_Default_Positions[i];
    Rpositions[i] = LR_Default_Positions[i];
    Mpositions[i] = M_Default_Positions[i];

    
  }

  Larm.moveServos(Lservos, 6, 1000);
  Rarm.moveServos(Rservos, 6, 1000);
  Marm.moveServos(Mservos, 6, 1000);

  while (Serial.available()) {
    Serial.read();
  }
}

void Move_Servos(int arm, int previous_positions[6], int new_positions[6], LobotServo servos[6]) {
  uint8_t changeMask = 0b00000000;
  int step[6];
  int distance[6];
  int maxDistance = 0;
  int iterations = 40;   // 100 loops × 10 ms ≈ 1 second

  // Determine which servos need movement
  for (int i = 0; i < 6; i++) {
    if (previous_positions[i] != new_positions[i]) {
      changeMask |= (1 << i);
    }
  }

  // Compute distances and track the largest
  for (int i = 0; i < 6; i++) {
    distance[i] = new_positions[i] - previous_positions[i];

    if (abs(distance[i]) > maxDistance) {
      maxDistance = abs(distance[i]);
      iterations = maxDistance / 100;
      if (iterations < 1) {
        iterations = 1;
      }
    }
  }

  // Compute dynamic step size for each servo
  for (int i = 0; i < 6; i++) {

    if (distance[i] == 0) {
      step[i] = 0;
    } else {

      step[i] = distance[i] / iterations;

      // Ensure at least 1 unit movement if needed
      if (step[i] == 0) {
        step[i] = (distance[i] > 0) ? 1 : -1;
      }
    }
  }

  while (changeMask != 0) {

    for (int index = 0; index < 6; index++) {

      if (changeMask & (1 << index)) {

        previous_positions[index] += step[index];

        // prevent overshoot
        if ((step[index] < 0) && (previous_positions[index] < new_positions[index])) {
          previous_positions[index] = new_positions[index];
        }
        else if ((step[index] > 0) && (previous_positions[index] > new_positions[index])) {
          previous_positions[index] = new_positions[index];
        }

        // check if finished
        if (previous_positions[index] == new_positions[index]) {
          changeMask &= ~(1 << index); // 	x ^= (1 << 2);
        }
      }
    }
    move(previous_positions, servos, arm);
  }
}

void move(int positions[6], LobotServo servos[6], int arm) { // This function takes the list of 6 positions and moves the servos using them; this is the low level function that replaces the hiwonder .moveServos function for my uses
  for (int i = 0; i < 6; i ++) {
    servos[i].Position = positions[i];
    //Serial.print("Servo "); Serial.print(i); Serial.print(" Position:"); Serial.print(servos[i].Position);
  }
  
  //Serial.print("; servo1 Position: "); Serial.print(servos[0].Position);
  //Serial.print("; servo2 Position: "); Serial.print(servos[1].Position);
  //Serial.print("; servo3 Position: "); Serial.print(servos[2].Position);
  //Serial.print("; servo4 Position: "); Serial.print(servos[3].Position);
  //Serial.print("; servo5 Position: "); Serial.print(servos[4].Position);
  //Serial.print("; servo6 Position: "); Serial.print(servos[5].Position);

  if (arm == 2) {
    Marm.moveServos(servos, 6, 1000);
  } else if (arm == 1) {
    Rarm.moveServos(servos, 6, 1000);
  } else if (arm == 0) {
    Larm.moveServos(servos, 6, 1000);
  }
  delay(10);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n'); // read full line
    int numValues = 7;
    int tempPositions[7];

    int index = 0;
    char* ptr = strtok((char*)line.c_str(), ",");
    while (ptr != NULL && index < numValues) {
        tempPositions[index++] = atoi(ptr);
        ptr = strtok(NULL, ",");
    }

    if (index == 7){
      for (int i = 0; i < 7; i++) {
        positions[i] = tempPositions[i];
      }

      int arm_identifier = positions[6]; // 2 is middle/transfer arm, 1 is right arm, 0 is left arm

      if (arm_identifier == 2) {
        Move_Servos(2, Mpositions, positions, Mservos);
      }
      else if (arm_identifier == 1) {
        Move_Servos(1, Rpositions, positions, Rservos);
      } 
      else if (arm_identifier == 0) {
        Move_Servos(0, Lpositions, positions, Lservos);
      } 

      Serial.print("[");
      for (int i = 0; i < 6; i++) {
        Serial.print(positions[i]);
        if (i < 5) {
          Serial.print(", ");
        }
      }
      Serial.println("]"); // Print the positions to serial so we receive it on computer end and can compare
    }
  }
}