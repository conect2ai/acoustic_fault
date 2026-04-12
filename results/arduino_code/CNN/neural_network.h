#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Arduino.h>

// --- Constantes da Arquitetura ---
// Gerado automaticamente a partir do modelo treinado

// Tamanho máximo do buffer necessário
#define MAX_BUFFER_SIZE 4440

// Tamanho da entrada (imagem)
#define INPUT_SIZE 663
#define INPUT_CHANNELS 1
#define INPUT_HEIGHT 17
#define INPUT_WIDTH 39

// Camada Convolucional 0
#define CONV0_IN_CHANNELS 1
#define CONV0_IN_H 17
#define CONV0_IN_W 39
#define CONV0_NUM_FILTERS 8
#define CONV0_KERNEL_SIZE 3
#define CONV0_OUT_H 15
#define CONV0_OUT_W 37

// Camada Pooling 0
#define POOL0_CHANNELS 8
#define POOL0_IN_H 15
#define POOL0_IN_W 37
#define POOL0_OUT_H 7
#define POOL0_OUT_W 18

// Camada Densa 0
#define DENSE0_INPUT_SIZE 1008
#define DENSE0_OUTPUT_SIZE 12

// Protótipos das funções
void setup_network();
float* predict(float* input_data);

#endif // NEURAL_NETWORK_H
