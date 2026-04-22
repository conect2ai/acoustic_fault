#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Arduino.h>

// --- Architecture Constants ---
// Automatically generated from trained model

// Maximum required buffer size
#define MAX_BUFFER_SIZE 2560

// Input image size
#define INPUT_SIZE 120
#define INPUT_CHANNELS 1
#define INPUT_HEIGHT 10
#define INPUT_WIDTH 12

// Convolutional Layer 0
#define CONV0_IN_CHANNELS 1
#define CONV0_IN_H 10
#define CONV0_IN_W 12
#define CONV0_NUM_FILTERS 32
#define CONV0_KERNEL_SIZE 3
#define CONV0_OUT_H 8
#define CONV0_OUT_W 10

// Pooling Layer 0
#define POOL0_CHANNELS 32
#define POOL0_IN_H 8
#define POOL0_IN_W 10
#define POOL0_OUT_H 4
#define POOL0_OUT_W 5

// Convolutional Layer 1
#define CONV1_IN_CHANNELS 32
#define CONV1_IN_H 4
#define CONV1_IN_W 5
#define CONV1_NUM_FILTERS 32
#define CONV1_KERNEL_SIZE 3
#define CONV1_OUT_H 2
#define CONV1_OUT_W 3

// Pooling Layer 1
#define POOL1_CHANNELS 32
#define POOL1_IN_H 2
#define POOL1_IN_W 3
#define POOL1_OUT_H 1
#define POOL1_OUT_W 1

// Dense Layer 0
#define DENSE0_INPUT_SIZE 32
#define DENSE0_OUTPUT_SIZE 128

// Dense Layer 1
#define DENSE1_INPUT_SIZE 128
#define DENSE1_OUTPUT_SIZE 12

// Function prototypes
void setup_network();
float* predict(float* input_data);

#endif // NEURAL_NETWORK_H
