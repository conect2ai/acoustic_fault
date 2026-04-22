#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Arduino.h>

// --- Architecture Constants ---
// Automatically generated from trained model

// Maximum required buffer size
#define MAX_BUFFER_SIZE 768

// Input image size
#define INPUT_SIZE 120
#define INPUT_CHANNELS 1
#define INPUT_HEIGHT 10
#define INPUT_WIDTH 12

// Convolutional Layer 0
#define CONV0_IN_CHANNELS 1
#define CONV0_IN_H 10
#define CONV0_IN_W 12
#define CONV0_NUM_FILTERS 16
#define CONV0_KERNEL_SIZE 5
#define CONV0_OUT_H 6
#define CONV0_OUT_W 8

// Pooling Layer 0
#define POOL0_CHANNELS 16
#define POOL0_IN_H 6
#define POOL0_IN_W 8
#define POOL0_OUT_H 3
#define POOL0_OUT_W 4

// Dense Layer 0
#define DENSE0_INPUT_SIZE 192
#define DENSE0_OUTPUT_SIZE 64

// Dense Layer 1
#define DENSE1_INPUT_SIZE 64
#define DENSE1_OUTPUT_SIZE 12

// Function prototypes
void setup_network();
float* predict(float* input_data);

#endif // NEURAL_NETWORK_H
