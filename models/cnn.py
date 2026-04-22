import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. PyTorch Model Definition
# -----------------------------------------------------------------------------

class CNNModel(nn.Module):
    """
    Simple CNN model for MNIST.
    Expects input shape (batch, 1, 28, 28).
    """
    def __init__(self, input_channels=1, num_classes=10):
        super(CNNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3),  # (batch, 8, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (batch, 8, 13, 13)
            nn.Flatten(),
            nn.Linear(8 * 13 * 13, num_classes)           # (batch, 10)
        )

    def forward(self, x):
        return self.layers(x)

# -----------------------------------------------------------------------------
# 2. Training and Validation Functions (using DataLoaders)
# -----------------------------------------------------------------------------

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# -----------------------------------------------------------------------------
# 3. JSON Export (original format)
# -----------------------------------------------------------------------------

def export_to_json(model, filename, input_shape=(1, 28, 28), device='cpu'):
    """
    Export the model architecture and weights to the JSON format
    expected by the C++ code generator.
    """
    model.eval()
    model.to('cpu')

    # Dummy forward to get intermediate shapes (used internally, not essential)
    dummy = torch.zeros(1, *input_shape)
    activations = []
    hooks = []

    def hook_fn(module, input, output):
        activations.append(output.shape)

    for layer in model.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(dummy)

    for hook in hooks:
        hook.remove()

    # Extract parametric layers (Conv2d and Linear)
    parametric_layers = []
    modules = list(model.layers)
    i = 0
    while i < len(modules):
        module = modules[i]
        if isinstance(module, nn.Conv2d):
            weights = module.weight.detach().cpu().numpy()
            biases = module.bias.detach().cpu().numpy()
            layer_info = {
                "type": "conv",
                "num_filters": weights.shape[0],
                "kernel_size": weights.shape[2],
                "weights": weights.tolist(),
                "biases": biases.tolist(),
                "activation": "linear"
            }
            # Check for ReLU after conv
            if i + 1 < len(modules) and isinstance(modules[i+1], nn.ReLU):
                layer_info["activation"] = "relu"
                i += 1
            # Check for MaxPool2d after conv (or after ReLU)
            if i + 1 < len(modules) and isinstance(modules[i+1], nn.MaxPool2d):
                pool = modules[i+1]
                layer_info["has_pooling"] = True
                layer_info["pool_size"] = pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[0]
                i += 1
            # Look ahead for Flatten to mark this conv as flattened
            j = i + 1
            while j < len(modules) and not isinstance(modules[j], (nn.Linear, nn.Conv2d)):
                if isinstance(modules[j], nn.Flatten):
                    layer_info["is_flattened"] = True
                    break
                j += 1
            parametric_layers.append(layer_info)

        elif isinstance(module, nn.Linear):
            weights = module.weight.detach().cpu().numpy()
            biases = module.bias.detach().cpu().numpy()
            input_size = weights.shape[1]
            output_size = weights.shape[0]
            layer_info = {
                "type": "dense",
                "input_size": input_size,
                "output_size": output_size,
                "weights": weights.tolist(),
                "biases": biases.tolist(),
                "activation": "linear"
            }
            # Check for ReLU after dense (except possibly last layer)
            if i + 1 < len(modules) and isinstance(modules[i+1], nn.ReLU):
                layer_info["activation"] = "relu"
                i += 1
            parametric_layers.append(layer_info)
        i += 1

    model_data = {
        "model_quantized": False,
        "num_layers": len(parametric_layers),
        "layers": parametric_layers
    }

    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=4)

    print(f"Model exported to {filename} with {len(parametric_layers)} parametric layers.")
    return model_data

# -----------------------------------------------------------------------------
# 4. C++ Code Generation Functions (identical to original)
# -----------------------------------------------------------------------------

def analyze_model_architecture(model_layers, input_shape):
    """
    Analyze the full model architecture from the layer list.
    input_shape: tuple (channels, height, width) of the input.
    """
    architecture = []
    current_shape = input_shape  # (channels, height, width)

    for i, layer in enumerate(model_layers):
        layer_info = {
            "original_index": i,
            "weights_shape": None,
            "biases_shape": None,
            "type": None,
            "activation": layer.get("activation", "linear")
        }

        # Determine layer type from weights dimensions
        weights = layer["weights"]

        if isinstance(weights[0][0], list) and isinstance(weights[0][0][0], list):
            # 4D: [num_filters][in_channels][kernel_h][kernel_w] -> Conv
            layer_info["type"] = "conv"
            layer_info["num_filters"] = len(weights)
            layer_info["in_channels"] = len(weights[0])
            layer_info["kernel_size"] = len(weights[0][0])  # assume square kernel

            # Compute output dimensions
            kernel_size = layer_info["kernel_size"]
            in_h, in_w = current_shape[1], current_shape[2]
            out_h = in_h - kernel_size + 1
            out_w = in_w - kernel_size + 1

            layer_info["input_shape"] = current_shape
            layer_info["output_shape"] = (layer_info["num_filters"], out_h, out_w)

            # Add convolutional layer to architecture
            architecture.append(layer_info)

            # Update current shape
            current_shape = layer_info["output_shape"]

            # Check for pooling after ReLU (indicated by "has_pooling" in JSON)
            if layer.get("has_pooling", False):
                pool_size = layer.get("pool_size", 2)
                pooled_h = out_h // pool_size
                pooled_w = out_w // pool_size

                pool_info = {
                    "type": "pool",
                    "pool_size": pool_size,
                    "input_shape": current_shape,
                    "output_shape": (layer_info["num_filters"], pooled_h, pooled_w)
                }
                architecture.append(pool_info)
                current_shape = pool_info["output_shape"]

        else:
            # 2D: [output_size][input_size] -> Dense
            layer_info["type"] = "dense"
            layer_info["output_size"] = len(weights)
            layer_info["input_size"] = len(weights[0])

            # For first dense layer after convolutions, compute flattened size
            if len(current_shape) == 3 and current_shape != (1, 1, 1):
                flattened_size = current_shape[0] * current_shape[1] * current_shape[2]
                layer_info["flattened_input_size"] = flattened_size

                # Add flatten operation
                flatten_info = {
                    "type": "flatten",
                    "input_shape": current_shape,
                    "output_size": flattened_size
                }
                architecture.append(flatten_info)

            layer_info["input_shape"] = current_shape
            layer_info["output_shape"] = (layer_info["output_size"], 1, 1)

            architecture.append(layer_info)
            current_shape = layer_info["output_shape"]

        # Record weight and bias shapes
        layer_info["weights_shape"] = get_shape_recursive(weights)
        layer_info["biases_shape"] = (len(layer["biases"]),)

    return architecture

def get_shape_recursive(arr):
    """Recursively obtain the shape of a nested list."""
    if not isinstance(arr, list):
        return ()
    shape = [len(arr)]
    if arr and isinstance(arr[0], list):
        shape.extend(get_shape_recursive(arr[0]))
    return tuple(shape)

def format_cpp_array(data, is_conv=False):
    """Format a Python list into a C++ array string."""
    def flatten(lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(item)
        return result

    flat_list = flatten(data)

    # Format into lines for readability
    s = "{\n    "
    for i, val in enumerate(flat_list):
        s += f"{val:.8f}f, "
        if (i + 1) % 8 == 0:  # 8 values per line
            s += "\n    "
    s = s.strip().strip(',') + "\n};"
    return s

def generate_arduino_code(json_path, output_dir=".", input_height=28, input_width=28):
    """
    Read the JSON file and generate .h and .cpp files for Arduino.
    input_height and input_width: dimensions of the input image (assumes 1 channel).
    """
    with open(json_path, 'r') as f:
        model = json.load(f)

    # Input shape: (channels=1, height, width)
    input_shape = (1, input_height, input_width)

    # Analyze model architecture
    architecture = analyze_model_architecture(model["layers"], input_shape)

    # --- Calculate MAX_BUFFER_SIZE ---
    buffer_sizes = []

    # Input size
    buffer_sizes.append(input_height * input_width)

    # Calculate intermediate sizes
    for layer in architecture:
        if layer["type"] in ["conv", "dense"]:
            if layer["type"] == "conv":
                # Convolution output: num_filters * out_h * out_w
                out_shape = layer["output_shape"]
                size = out_shape[0] * out_shape[1] * out_shape[2]
            else:  # dense
                # Dense output: output_size
                size = layer["output_size"]
            buffer_sizes.append(size)
        elif layer["type"] == "pool":
            out_shape = layer["output_shape"]
            size = out_shape[0] * out_shape[1] * out_shape[2]
            buffer_sizes.append(size)
        elif layer["type"] == "flatten":
            size = layer["output_size"]
            buffer_sizes.append(size)

    max_buffer_size = max(buffer_sizes)

    # --- Generate .h file ---
    h_content = '''#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Arduino.h>

// --- Architecture Constants ---
// Automatically generated from trained model

'''

    # General constants
    h_content += f"""// Maximum required buffer size
#define MAX_BUFFER_SIZE {max_buffer_size}

// Input image size
#define INPUT_SIZE {input_height * input_width}
#define INPUT_CHANNELS 1
#define INPUT_HEIGHT {input_height}
#define INPUT_WIDTH {input_width}

"""

    # Counters for layer naming
    conv_idx = 0
    dense_idx = 0
    pool_idx = 0

    # Generate constants for each layer
    for layer in architecture:
        if layer["type"] == "conv":
            in_channels, in_h, in_w = layer["input_shape"]
            out_channels, out_h, out_w = layer["output_shape"]

            h_content += f"// Convolutional Layer {conv_idx}\n"
            h_content += f"#define CONV{conv_idx}_IN_CHANNELS {in_channels}\n"
            h_content += f"#define CONV{conv_idx}_IN_H {in_h}\n"
            h_content += f"#define CONV{conv_idx}_IN_W {in_w}\n"
            h_content += f"#define CONV{conv_idx}_NUM_FILTERS {out_channels}\n"
            h_content += f"#define CONV{conv_idx}_KERNEL_SIZE {layer['kernel_size']}\n"
            h_content += f"#define CONV{conv_idx}_OUT_H {out_h}\n"
            h_content += f"#define CONV{conv_idx}_OUT_W {out_w}\n\n"

            layer["c_name"] = f"conv{conv_idx}"
            conv_idx += 1

        elif layer["type"] == "dense":
            if "flattened_input_size" in layer:
                input_size = layer["flattened_input_size"]
            else:
                input_size = layer["input_size"]

            h_content += f"// Dense Layer {dense_idx}\n"
            h_content += f"#define DENSE{dense_idx}_INPUT_SIZE {input_size}\n"
            h_content += f"#define DENSE{dense_idx}_OUTPUT_SIZE {layer['output_size']}\n\n"

            layer["c_name"] = f"dense{dense_idx}"
            dense_idx += 1

        elif layer["type"] == "pool":
            in_channels, in_h, in_w = layer["input_shape"]
            out_channels, out_h, out_w = layer["output_shape"]

            h_content += f"// Pooling Layer {pool_idx}\n"
            h_content += f"#define POOL{pool_idx}_CHANNELS {in_channels}\n"
            h_content += f"#define POOL{pool_idx}_IN_H {in_h}\n"
            h_content += f"#define POOL{pool_idx}_IN_W {in_w}\n"
            h_content += f"#define POOL{pool_idx}_OUT_H {out_h}\n"
            h_content += f"#define POOL{pool_idx}_OUT_W {out_w}\n\n"

            layer["c_name"] = f"pool{pool_idx}"
            pool_idx += 1

    h_content += '''// Function prototypes
void setup_network();
float* predict(float* input_data);

#endif // NEURAL_NETWORK_H
'''

    # --- Generate .cpp file ---
    cpp_content = '''#include "neural_network.h"

// --- Global Buffers ---
static float buffer1[MAX_BUFFER_SIZE];
static float buffer2[MAX_BUFFER_SIZE];

'''

    # Add weight and bias arrays
    for layer in architecture:
        if layer["type"] == "conv":
            orig_layer = model["layers"][layer["original_index"]]
            cpp_content += f"// Weights for convolutional layer {layer['c_name']}\n"
            cpp_content += f"const float {layer['c_name']}_weights[] PROGMEM = {format_cpp_array(orig_layer['weights'], is_conv=True)};\n\n"
            cpp_content += f"// Biases for convolutional layer {layer['c_name']}\n"
            cpp_content += f"const float {layer['c_name']}_biases[] PROGMEM = {format_cpp_array(orig_layer['biases'])};\n\n"

        elif layer["type"] == "dense":
            orig_layer = model["layers"][layer["original_index"]]
            cpp_content += f"// Weights for dense layer {layer['c_name']}\n"
            cpp_content += f"const float {layer['c_name']}_weights[] PROGMEM = {format_cpp_array(orig_layer['weights'])};\n\n"
            cpp_content += f"// Biases for dense layer {layer['c_name']}\n"
            cpp_content += f"const float {layer['c_name']}_biases[] PROGMEM = {format_cpp_array(orig_layer['biases'])};\n\n"

    # --- Helper functions ---
    cpp_content += '''// Helper functions

void relu(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] < 0) {
            arr[i] = 0;
        }
    }
}

void softmax(float* arr, int size) {
    float max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    
    float sum_exp = 0.0;
    for (int i = 0; i < size; i++) {
        arr[i] = exp(arr[i] - max_val);
        sum_exp += arr[i];
    }
    
    for (int i = 0; i < size; i++) {
        arr[i] /= sum_exp;
    }
}

void dense_layer(float* output, const float* input, int input_size, int output_size, 
                 const float* weights, const float* biases) {
    for (int i = 0; i < output_size; i++) {
        float sum = 0;
        for (int j = 0; j < input_size; j++) {
            #if defined(__AVR__)
            float weight = pgm_read_float_near(weights + i * input_size + j);
            #else
            float weight = weights[i * input_size + j];
            #endif
            sum += weight * input[j];
        }
        #if defined(__AVR__)
        output[i] = sum + pgm_read_float_near(biases + i);
        #else
        output[i] = sum + biases[i];
        #endif
    }
}

void conv_layer(float* output, const float* input, int in_channels, int in_h, int in_w,
                int out_channels, int kernel_size, const float* kernels, const float* biases) {
    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;
    
    for (int f = 0; f < out_channels; f++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float sum = 0;
                for (int c = 0; c < in_channels; c++) {
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            int input_y = y + ky;
                            int input_x = x + kx;
                            int kernel_idx = f * (in_channels * kernel_size * kernel_size) + 
                                           c * (kernel_size * kernel_size) + 
                                           ky * kernel_size + kx;
                            int input_idx = c * (in_h * in_w) + input_y * in_w + input_x;
                            
                            #if defined(__AVR__)
                            float weight = pgm_read_float_near(kernels + kernel_idx);
                            #else
                            float weight = kernels[kernel_idx];
                            #endif
                            sum += input[input_idx] * weight;
                        }
                    }
                }
                #if defined(__AVR__)
                output[f * (out_h * out_w) + y * out_w + x] = sum + pgm_read_float_near(biases + f);
                #else
                output[f * (out_h * out_w) + y * out_w + x] = sum + biases[f];
                #endif
            }
        }
    }
}

void max_pool_2x2(float* output, const float* input, int channels, int in_h, int in_w) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float max_val = -1e9;
                for (int pool_y = 0; pool_y < 2; pool_y++) {
                    for (int pool_x = 0; pool_x < 2; pool_x++) {
                        int input_idx = c * (in_h * in_w) + 
                                       (y * 2 + pool_y) * in_w + 
                                       (x * 2 + pool_x);
                        float val = input[input_idx];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                output[c * (out_h * out_w) + y * out_w + x] = max_val;
            }
        }
    }
}

// --- Main Inference Function ---
'''

    # Generate predict() function calls
    cpp_content += '''
void setup_network() {
    // No initialization needed for this implementation
}

float* predict(float* input_data) {
    float* buffer_a = buffer1;
    float* buffer_b = buffer2;
    float* temp_ptr = nullptr;
    
    // Copy input data
    for (int i = 0; i < INPUT_SIZE; i++) {
        buffer_a[i] = input_data[i];
    }
'''

    conv_idx = 0
    dense_idx = 0
    pool_idx = 0

    for layer in architecture:
        if layer["type"] == "conv":
            cpp_content += f'''
    // Convolutional Layer {conv_idx}
    conv_layer(buffer_b, buffer_a, 
               CONV{conv_idx}_IN_CHANNELS, CONV{conv_idx}_IN_H, CONV{conv_idx}_IN_W,
               CONV{conv_idx}_NUM_FILTERS, CONV{conv_idx}_KERNEL_SIZE,
               {layer['c_name']}_weights, {layer['c_name']}_biases);
    
    // Swap buffers
    temp_ptr = buffer_a;
    buffer_a = buffer_b;
    buffer_b = temp_ptr;
'''
            if layer.get("activation") == "relu":
                size = layer["output_shape"][0] * layer["output_shape"][1] * layer["output_shape"][2]
                cpp_content += f'''
    // ReLU
    relu(buffer_a, {size});
'''
            conv_idx += 1

        elif layer["type"] == "pool":
            cpp_content += f'''
    // MaxPool {layer['pool_size']}x{layer['pool_size']}
    max_pool_2x2(buffer_b, buffer_a,
                 POOL{pool_idx}_CHANNELS, POOL{pool_idx}_IN_H, POOL{pool_idx}_IN_W);
    
    // Swap buffers
    temp_ptr = buffer_a;
    buffer_a = buffer_b;
    buffer_b = temp_ptr;
'''
            pool_idx += 1

        elif layer["type"] == "flatten":
            cpp_content += '''
    // Flatten (implicit - data already in buffer_a as 1D vector)
'''

        elif layer["type"] == "dense":
            cpp_content += f'''
    // Dense Layer {dense_idx}
    dense_layer(buffer_b, buffer_a,
                DENSE{dense_idx}_INPUT_SIZE, DENSE{dense_idx}_OUTPUT_SIZE,
                {layer['c_name']}_weights, {layer['c_name']}_biases);
    
    // Swap buffers
    temp_ptr = buffer_a;
    buffer_a = buffer_b;
    buffer_b = temp_ptr;
'''
            if layer.get("activation") == "relu" and layer != architecture[-1]:
                cpp_content += f'''
    // ReLU
    relu(buffer_a, DENSE{dense_idx}_OUTPUT_SIZE);
'''
            if layer == architecture[-1]:
                cpp_content += f'''
    // Softmax on final layer
    softmax(buffer_a, DENSE{dense_idx}_OUTPUT_SIZE);
'''
            dense_idx += 1

    cpp_content += '''
    
    return buffer_a;
}
'''

    # Save files
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "neural_network.h"), "w") as f:
        f.write(h_content)

    with open(os.path.join(output_dir, "neural_network.cpp"), "w") as f:
        f.write(cpp_content)

    # Print info
    print(f"✓ Files generated in '{output_dir}':")
    print(f"  - neural_network.h")
    print(f"  - neural_network.cpp")
    print(f"\n📊 Automatically detected architecture:")
    print(f"  MAX_BUFFER_SIZE computed: {max_buffer_size}")
    print(f"  Total layers in architecture: {len(architecture)}")

    for i, layer in enumerate(architecture):
        if layer["type"] == "conv":
            print(f"  [{i}] Conv: {layer['input_shape'][0]}→{layer['output_shape'][0]} filters")
        elif layer["type"] == "dense":
            print(f"  [{i}] Dense: {layer.get('flattened_input_size', layer['input_size'])}→{layer['output_size']} neurons")
        elif layer["type"] == "pool":
            print(f"  [{i}] Pool: {layer['input_shape'][1]}x{layer['input_shape'][2]} → {layer['output_shape'][1]}x{layer['output_shape'][2]}")
        elif layer["type"] == "flatten":
            print(f"  [{i}] Flatten: {layer['input_shape'][0]}x{layer['input_shape'][1]}x{layer['input_shape'][2]} → {layer['output_size']}")

# -----------------------------------------------------------------------------
# 5. Main training function that accepts data arrays
# -----------------------------------------------------------------------------

def train_model(X_train, y_train, X_val, y_val,
                epochs=5,
                batch_size=64,
                learning_rate=0.001,
                device=None,
                seed=42,
                save_json="model_architecture.json",
                input_shape=(1, 28, 28)):
    """
    Train the CNN model with provided data.

    Parameters:
        X_train : np.ndarray
            Training data. Shape can be (N, 28, 28) or (N, 1, 28, 28).
            Values should be normalized (e.g., between 0 and 1).
        y_train : np.ndarray
            Training labels, shape (N,) with integers (0-9).
        X_val : np.ndarray
            Validation data, same format as X_train.
        y_val : np.ndarray
            Validation labels.
        epochs : int
            Number of epochs.
        batch_size : int
            Batch size.
        learning_rate : float
            Learning rate.
        device : str or None
            Device ('cuda', 'cpu'). If None, auto-detection.
        seed : int
            Random seed for reproducibility.
        save_json : str
            Path to save the JSON model file.
        input_shape : tuple
            Expected input shape (channels, height, width). Used for reference.
    """
    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    # Seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Convert data to tensors and adjust dimensions
    def prepare_data(X, y):
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        # Add channel dimension if necessary (assumes grayscale)
        if X.ndim == 3:          # (N, H, W)
            X = X[:, np.newaxis, :, :]  # (N, 1, H, W)
        elif X.ndim == 2:         # (N, H*W) flattened
            raise ValueError("X must have 3 dimensions (N, H, W) or 4 (N, C, H, W).")
        # If already 4D, assume second dimension is channels

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        return X_tensor, y_tensor

    X_train_t, y_train_t = prepare_data(X_train, y_train)
    X_val_t, y_val_t = prepare_data(X_val, y_val)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, criterion, optimizer
    model = CNNModel(input_channels=input_shape[0], num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")

    print(f"\nTraining completed. Best accuracy: {best_acc:.4f}")

    # Export to JSON
    model.load_state_dict(torch.load("best_model.pth"))
    model.to('cpu')
    export_to_json(model, save_json, input_shape=input_shape)

    # Generate Arduino code (optional, uncomment if desired)
    generate_arduino_code(save_json, output_dir="arduino_code",
                          input_height=input_shape[1], input_width=input_shape[2])

    return model