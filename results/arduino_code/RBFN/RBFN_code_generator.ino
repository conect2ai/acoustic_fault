#include "RbfClassificationModelRBF.h"

// Number of input features
#define INPUT_SIZE 120

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

// Instantiate the RBF classification model
RbfClassificationModelRBF model;

// Arduino setup function (executed once)
void setup() {
  // Initialize serial communication
  Serial.begin(115200);

  // Wait until the serial interface is ready (important for some boards)
  while (!Serial) {
    ; // Blocking wait
  }

  // Informational message
  Serial.println("Initializing the RBF neural network...");

  // Print model information (e.g., architecture, parameters)
  model.printInfo();
  
  Serial.println("Performing inference...");

  // Array to store output probabilities (number of classes defined by the model)
  float predictions[RBF_OUTPUT_DIM];

  // Perform inference using the RBF model
  // Input: feature vector
  // Output: probability distribution across classes
  model.predict(input, predictions);

  // Display output probabilities
  Serial.println("Output probabilities (classes 0 to N-1):");

  // Variables to determine the predicted class (argmax)
  int predicted_class = -1;
  float max_prob = -1.0;

  // Iterate over all output classes
  for (int i = 0; i < RBF_OUTPUT_DIM; i++) {
    Serial.print("Class ");
    Serial.print(i);
    Serial.print(": ");
    
    // Print probability with 6 decimal precision
    Serial.println(predictions[i], 6);

    // Track the class with the highest probability
    if (predictions[i] > max_prob) {
      max_prob = predictions[i];
      predicted_class = i;
    }
  }

  Serial.println("--------------------");

  // Print final predicted class
  Serial.print("Predicted class: ");
  Serial.println(predicted_class);

  // Print associated confidence (maximum probability)
  Serial.print("With probability: ");
  Serial.println(max_prob, 6);

  Serial.println("--------------------");
}

// Arduino loop function (not used in this example)
void loop() {
  // No repeated execution is required.
  // Inference is performed once during setup.
}