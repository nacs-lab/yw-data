const int analogInPin = A0;  // Analog input pin - FP signal
const int analogInPin_ref = A4;  // Analog input pin - FP Drive
const int analogOutPin = 4;  // Analog output pin

const double maximum = 4;  // Maximum voltage readings
const double threshold = maximum * 0.90;  // Threshold when the recovery happens
double current = 125;  // Initial output in the range between 0 to 255
int v = 0;   // A variable that help sweeping through the output
const int N = 1;  // Maximum data storaged

void setup() {
  Serial.begin(115200); // set baudrate
  Serial.println(); // test print
}

/* Original read, dosen't store, keep track of the maximum */
double read() {
  double sum = 0;
  int totalRun = 1;
  for (int i = 0; i < totalRun; i ++){
    int max = 0;
    unsigned long previousTime = millis();
    // 500 ms reading
    while (millis() - previousTime < 500) {
      int temp = analogRead(A0);
      if (temp > max){
        max = temp;
      }
    }
    sum = sum + max;
  }
  double rst = sum * 5 / 1023 / totalRun;
  return rst;
}

/* Read that store the data in an array, calc the maximum in the end */
int read_alt(){
  int max = 0;
  unsigned int arr[N];
  unsigned long previousTime = millis();

  int j = -1;
  while (millis() - previousTime < 200) {
    j = j + 1;
    arr[j] = analogRead(A0);
  }
  
  for (int k = 0; k < j; k++){
    Serial.println("Min:0,Max:1023");
    Serial.println(arr[k]);
    if (arr[k] > max){
      max = arr[k];
    }
  }
  //Serial.print("j = ");
  //Serial.println(j);
  return max;
}

/* Reading with a reference to the drive */
int read_ref(){
  int sum = 0;
  int lower = 950; // time to start the reading
  int upper = 1024; // time to end the reading
  unsigned long previousTime = millis();
  while(millis() - previousTime < 500){
    int temp = analogRead(analogInPin_ref);
    Serial.print("RefRead: ");
    Serial.println(temp);
    if(temp>lower && temp<upper){
      int read = analogRead(analogInPin);
      Serial.print("TempRead: ");
      Serial.println(read);
      sum = sum + read;
    }
  }
  return sum;
}

/*------------ADC setup------------*/

const byte adcPin = 0;  // A0
volatile int results [N];
volatile int resultNumber;
volatile int MaxRead;

// ADC complete ISR
ISR (ADC_vect){
  /* Running maximum */
  int temp = ADC;
  resultNumber ++;
  if (temp > MaxRead){
    MaxRead = temp;
  }
  /* store in an array reading */
  /*
  if (resultNumber >= MAX_RESULTS)
    ADCSRA = 0;  // turn off ADC
  else
    results [resultNumber++] = ADC;
  */
} // end of ADC_vect
  
EMPTY_INTERRUPT (TIMER1_COMPB_vect);

/* Fast reading with ADC */
double read_ADC(){
  unsigned long previousTime = millis();
  
  /* Reading in ~ 60 kHz */
  // reset Timer 1
  TCCR1A  = 0;
  TCCR1B  = 0;
  TCNT1   = 0;
  TCCR1B  = bit (CS11) | bit (WGM12);  // CTC, prescaler of 8
  TIMSK1  = bit (OCIE1B); 
  OCR1A   = 39;    
  OCR1B   = 39; // 20 uS - sampling frequency 50 kHz
  ADCSRA  =  bit (ADEN) | bit (ADIE) | bit (ADIF); // turn ADC on, want interrupt on completion
  ADCSRA |= bit (ADPS2);  // Prescaler of 16
  ADMUX   = bit (REFS0) | (adcPin & 7);
  ADCSRB  = bit (ADTS0) | bit (ADTS2);  // Timer/Counter1 Compare Match B
  ADCSRA |= bit (ADATE);   // turn on automatic triggering
  
  /* Reading in ~ 150 kHz */
  /*
  ADCSRA = 0;             // clear ADCSRA register
  ADCSRB = 0;             // clear ADCSRB register
  ADMUX |= (0 & 0x07);    // set A0 analog input pin
  ADMUX |= (1 << REFS0);  // set reference voltage
  ADMUX |= (1 << ADLAR);  // left align ADC value to 8 bits from ADCH register

  // sampling rate is [ADC clock] / [prescaler] / [conversion clock cycles]
  // for Arduino Uno ADC clock is 16 MHz and a conversion takes 13 clock cycles
  // ADCSRA |= (1 << ADPS2) | (1 << ADPS0);    // 32 prescaler for 38.5 KHz
  //ADCSRA |= (1 << ADPS2);                     // 16 prescaler for 76.9 KHz
  ADCSRA |= (1 << ADPS1) | (1 << ADPS0);    // 8 prescaler for 153.8 KHz

  ADCSRA |= (1 << ADATE); // enable auto trigger
  ADCSRA |= (1 << ADIE);  // enable interrupts when measurement complete
  ADCSRA |= (1 << ADEN);  // enable ADC
  ADCSRA |= (1 << ADSC);  // start ADC measurements
  */

  while (millis() - previousTime < 200){
    // Keep reading through ADC
    // Serial.println (results [i]);
  }
  ADCSRA = 0; //turn ADC off

  // Find out the maximum
  /*
  double max = 0;
  for (int k = 0; k < resultNumber; k++){
    //Serial.println("Min:0,Max:1023");
    //Serial.println(results[k]);
    if (results[k] > max){
      max = results[k];
    }
  }
  */
  Serial.print("count = ");
  Serial.println(resultNumber);

  resultNumber = 0; // reset counter
  double rst = MaxRead * 5 / 1023;
  Serial.print("MaxRead = ");
  Serial.println(MaxRead);
  return MaxRead;
}


/* Set the output within a range */
int set(int outputValue) {
  // 1 step ~ 0.02mA
  // int outputValue = map(current, 0, 100, 0, 255);
  if (outputValue > 255){
    outputValue = 255;
  }
  if (outputValue < 0){
    outputValue = 0;
  }
  analogWrite(analogOutPin, outputValue);
  delay(500);
  return outputValue;
}

/* Active servo to keep the postion near the right */
void active_servo() {
  double delta = 1;    // 0.02mA
  double p1 = 0.005;   // 0.5%
  double p2 = 0.01;    // 1%
  double ref = read();
  Serial.print("ref read is ");
  Serial.println(ref);
  set(current + delta);
  double plus = read();
  if (ref - plus < p1 * ref) { // Try if a slightly higher current won't impact so much
    current = current + delta;
    Serial.print("plus read is ");
    Serial.println(plus);
    Serial.print("active_servo_increase to ");
    Serial.println(current);
  } else { // Higher current unlocks the laser, try lower
    set(current - delta);
    double minus = read();
    if (minus - ref > p2 * ref) { // Lower current makes the lock significantly better
      current = current - delta;
      Serial.print("minus read is ");
      Serial.println(minus);
      Serial.print("active_servo_decrease to ");
      Serial.println(current);
    } else { // Neither higher or lower current is better, remain the current status
      set(current);
      Serial.print("remain read is ");
      Serial.println(minus);
      Serial.print("active_servo_remain at ");
      Serial.println(current);
    }
  }
}

/* Recovery that first jump to the right and gradually gets low */
void recovery() {
  current = current + 51; // Jumps 51 * 0.02mA ~ 1mA
  current = set(current);
  Serial.print("recovery_increase: ");
  Serial.println(current);
  double readValue = read();
  while (readValue < threshold) {
    current = current - 5; // Decrease 5 * 0.02mA ~ 0.1mA
    current = set(current);
    Serial.print("Reading is ");
    Serial.println(readValue);
    Serial.print("recovery_decrease: ");
    Serial.println(current);
    readValue = read();
  }
}

void loop() {
  


  /*
  // main loop
  if (read_ADC() > threshold){
    active_servo();
  }
  else{
    recovery();
  }
  */

  
  // check reading
  double x = 0;
  for (int i = 0; i < 2; i ++)
    x = x+ read_ADC();
  //analogRead(A0);
  x=x/10;
  Serial.print("Read: ");
  Serial.println(x);
  
  
  // check the output
  /*
  v=v+10;
  v=v%255;
  set(200);
  Serial.println(v);
  */

}