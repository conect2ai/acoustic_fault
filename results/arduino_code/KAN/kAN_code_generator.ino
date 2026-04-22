#include "kan_model.h"

// Number of input features
#define INPUT_SIZE 120

// Number of inference runs for benchmarking
#define NUM_RUNS 10

// Input feature vector (preprocessed sample)
float input[INPUT_SIZE] = {
    5.922186374664307,
    -4.324369430541992,
    -8.462364196777344,
    -12.641783714294434,
    -14.989919662475586,
    -17.428586959838867,
    -18.9476375579834,
    -17.287843704223633,
    -17.642295837402344,
    -19.6263370513916,
    -19.309566497802734,
    -20.470748901367188,
    -21.07651138305664,
    -21.017982482910156,
    -19.943641662597656,
    -20.134105682373047,
    -20.922216415405273,
    -21.372644424438477,
    -22.643959045410156,
    -24.88400650024414,
    -24.095199584960938,
    -23.376392364501953,
    -23.0485897064209,
    -24.296472549438477,
    -26.990890502929688,
    -28.53644561767578,
    -28.737096786499023,
    -29.233728408813477,
    -29.603487014770508,
    -28.8476505279541,
    2.9609086513519287,
    3.5317087173461914,
    3.386260509490967,
    4.149324417114258,
    3.6132500171661377,
    3.1774954795837402,
    3.912832260131836,
    4.33785343170166,
    3.97550106048584,
    3.3986077308654785,
    3.6414849758148193,
    3.567225456237793,
    3.817584276199341,
    3.700486660003662,
    3.56339693069458,
    3.235038995742798,
    3.149056911468506,
    3.205322742462158,
    2.991994857788086,
    3.0833184719085693,
    2.703706979751587,
    2.761458158493042,
    2.7861740589141846,
    2.383594274520874,
    2.5278103351593018,
    2.298642873764038,
    2.408348321914673,
    2.103238821029663,
    2.1561930179595947,
    2.3028712272644043,
    -0.0009165735100395977,
    0.015078498050570488,
    -0.01131165400147438,
    -0.06368035078048706,
    0.025812463834881783,
    -0.04851869121193886,
    -0.09998814016580582,
    -0.04268302768468857,
    -0.023275740444660187,
    -0.07588347792625427,
    0.014035765081644058,
    0.01969206891953945,
    -0.12232267111539841,
    -0.08317793160676956,
    -0.07838603854179382,
    -0.09463519603013992,
    0.01398052554577589,
    -0.03609450161457062,
    -0.05198933184146881,
    -0.002386815380305052,
    -0.02785995975136757,
    0.011771152727305889,
    -0.055712420493364334,
    -0.08136770129203796,
    -0.008295270614326,
    0.021997803822159767,
    0.033428240567445755,
    -0.012807807885110378,
    -0.046474333852529526,
    -0.05318339169025421,
    0.29608795046806335,
    0.47789502143859863,
    0.3992801606655121,
    0.6192410588264465,
    0.36708492040634155,
    0.5396157503128052,
    0.5733515024185181,
    0.5509222149848938,
    0.5538081526756287,
    0.40036270022392273,
    0.45095667243003845,
    0.4297320246696472,
    0.5983821749687195,
    0.5677046775817871,
    0.3804335594177246,
    0.42989465594291687,
    0.31589192152023315,
    0.460214227437973,
    0.4485626220703125,
    0.4397154748439789,
    0.624186635017395,
    0.4410053789615631,
    0.33400383591651917,
    0.3454917371273041,
    0.3698458671569824,
    0.2796242833137512,
    0.46625033020973206,
    0.3940142095088959,
    0.34271642565727234,
    0.3703310787677765};

// Arduino setup function (executed once)
void setup()
{
  // Initialize serial communication
  Serial.begin(115200);

  // Delay to ensure stable initialization
  delay(2000);

  // Informational message
  Serial.println("Running KAN inference benchmark...");

  // Array to store inference execution times
  unsigned long times[NUM_RUNS];

  // Variable to store predicted class
  int predicted_class;

  // Perform multiple inference runs
  for (int i = 0; i < NUM_RUNS; i++)
  {
    // Record start time (microseconds)
    unsigned long start_time = micros();

    // Perform inference using the KAN model
    // Note: The model expects each feature as an individual argument
    predicted_class = predict(
        input[0], input[1], input[2], input[3],
        input[4], input[5], input[6], input[7],
        input[8], input[9], input[10], input[11],
        input[12], input[13], input[14], input[15],
        input[16], input[17], input[18], input[19],
        input[20], input[21], input[22], input[23],
        input[24], input[25], input[26], input[27],
        input[28], input[29], input[30], input[31],
        input[32], input[33], input[34], input[35],
        input[36], input[37], input[38], input[39],
        input[40], input[41], input[42], input[43],
        input[44], input[45], input[46], input[47],
        input[48], input[49], input[50], input[51],
        input[52], input[53], input[54], input[55],
        input[56], input[57], input[58], input[59],
        input[60], input[61], input[62], input[63],
        input[64], input[65], input[66], input[67],
        input[68], input[69], input[70], input[71],
        input[72], input[73], input[74], input[75],
        input[76], input[77], input[78], input[79],
        input[80], input[81], input[82], input[83],
        input[84], input[85], input[86], input[87],
        input[88], input[89], input[90], input[91],
        input[92], input[93], input[94], input[95],
        input[96], input[97], input[98], input[99],
        input[100], input[101], input[102], input[103],
        input[104], input[105], input[106], input[107],
        input[108], input[109], input[110], input[111],
        input[112], input[113], input[114], input[115],
        input[116], input[117], input[118], input[119]);

    // Record end time
    unsigned long end_time = micros();

    // Store execution time
    times[i] = end_time - start_time;
  }

  // -------------------------------------------------
  // Compute average inference time (excluding first run)
  // The first run is typically affected by initialization overhead
  // -------------------------------------------------
  unsigned long sum = 0;

  for (int i = 1; i < NUM_RUNS; i++)
  {
    sum += times[i];
  }

  float avg_time = (float)sum / (NUM_RUNS - 1);

  Serial.println("--------------------");

  // Print average inference time in microseconds
  Serial.print("Average time: ");
  Serial.print(avg_time);
  Serial.println(" us");

  // Print average inference time in milliseconds
  Serial.print("Average time: ");
  Serial.print(avg_time / 1000.0);
  Serial.println(" ms");

  // Print predicted class from the last inference
  Serial.print("Predicted class: ");
  Serial.println(predicted_class);

  Serial.println("--------------------");
}

// Arduino loop function (unused in this benchmark)
void loop()
{
}