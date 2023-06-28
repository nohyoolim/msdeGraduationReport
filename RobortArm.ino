#include <Servo.h>
Servo myservo1;
Servo myservo2;
Servo myservo3;
Servo myservo4;
Servo myservo5;
Servo myservo6;

String incomingByte;
int slashPosition;
int kPosition;

void setup() {
  Serial.begin(9600);

  myservo1.attach(2);
  myservo2.attach(4);
  myservo3.attach(6);
  myservo4.attach(8);
  myservo5.attach(10);
  myservo6.attach(11);
  
}

 
const int BUFFER_SIZE = 100;
char buf[BUFFER_SIZE];

void loop() {
  if(Serial.available()>0){

    int rlen = Serial.readBytesUntil('k',buf,BUFFER_SIZE);      //여러 개의 각도를 한번에 받습니다. 마지막에는 'k'를 붙입니다.
    incomingByte=buf;                                 //각도들은 '/'로 구분합니다.

    slashPosition = incomingByte.indexOf('/');

    myservo1.write(incomingByte.substring(0,slashPosition).toInt());    //처음 '/'가 있는 부분까지 문자열을 분리해서 서보모터에 입력하는 코드
    incomingByte=incomingByte.substring(slashPosition+1,incomingByte.length());    
    delay(300);

    slashPosition = incomingByte.indexOf('/');

    myservo2.write(incomingByte.substring(0,slashPosition).toInt());
    incomingByte=incomingByte.substring(slashPosition+1,incomingByte.length());
    delay(300);

    slashPosition = incomingByte.indexOf('/');

    myservo3.write(incomingByte.substring(0,slashPosition).toInt());
    incomingByte=incomingByte.substring(slashPosition+1,incomingByte.length());
    delay(300);

    slashPosition = incomingByte.indexOf('/');


    myservo4.write(incomingByte.substring(0,slashPosition).toInt());
    incomingByte=incomingByte.substring(slashPosition+1,incomingByte.length());
    delay(300);

    slashPosition = incomingByte.indexOf('/');


    myservo5.write(incomingByte.substring(0,slashPosition).toInt());
    incomingByte=incomingByte.substring(slashPosition+1,incomingByte.length());
    delay(300);


    myservo6.write(incomingByte.toInt());
    delay(300);

    Serial.end();
    Serial.begin(9600);
  }
}