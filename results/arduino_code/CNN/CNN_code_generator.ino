#include "neural_network.h"

// --- Exemplo de uma imagem de entrada (MNIST 28x28) ---
// Em uma aplicação real, você obteria isso de um sensor ou da memória.
// Este é um exemplo achatado (28*28 = 784 pixels).
// Para este teste, vamos usar dados aleatórios normalizados entre 0 e 1.
float sample_image[INPUT_SIZE];


void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Espera a porta serial conectar
  }

  Serial.println("Inicializando a rede neural...");
  setup_network();
  
float sample_image[120] = {

};


  Serial.println("Realizando inferência...");

  // Chama a função de predição
  float* predictions = predict(sample_image);

  // Imprime as probabilidades de saída
  Serial.println("Probabilidades de saída (classes 0-9):");
  int predicted_class = -1;
  float max_prob = -1.0;

  for (int i = 0; i < 12; i++) { // Assumindo 10 classes de saída para MNIST
    Serial.print("Classe ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(predictions[i], 6); // Imprime com 6 casas decimais

    if (predictions[i] > max_prob) {
      max_prob = predictions[i];
      predicted_class = i;
    }
  }

  Serial.println("--------------------");
  Serial.print("Classe prevista: ");
  Serial.println(predicted_class);
  Serial.print("Com probabilidade de: ");
  Serial.println(max_prob, 6);
  Serial.println("--------------------");
}

void loop() {
  // Nada a fazer no loop, a inferência foi feita no setup.
}