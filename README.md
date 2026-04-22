&nbsp;
&nbsp;
<p align="center">
  <img width="800" src="./figures/conecta_logo.png" />
</p> 
&nbsp;

# Acoustic Fault Detection in Combustion Engines Under TinyML Constraints

### ✍🏾 Authors
[Thommas K. S. Flores](https://github.com/thommaskevin),  [João Carlos N. Bittencourt](https://github.com/),  [Thiago C. Jesus](https://github.com/),   [Daniel G. Costa](https://github.com/daniel-gcosta) and  [Ivanovitch Silva](https://github.com/ivanovitchm)

---

This repository provides the **code, datasets, and experimental resources** supporting the paper:

> **Acoustic Fault Detection in Combustion Engines Under TinyML Constraints**  
> *Published in ACM Transactions on Embedded Computing Systems (TECS)*

The work presents a **hardware‑aware evaluation** of five lightweight neural architectures for ignition fault detection in internal combustion engines using acoustic signals.  
We compare **Fourier Analysis Networks (FAN)**, **Convolutional Neural Networks (CNN)**, **Multi‑Layer Perceptrons (MLP)**, **Kolmogorov‑Arnold Networks (KAN)**, and **Radial Basis Function Networks (RBFN)** under TinyML constraints.  
The analysis spans **nine commercial microcontrollers** and two acoustic feature representations (**MFCC** and **MFE**), measuring accuracy, memory footprint, inference latency, and energy consumption.

--



## 📁 Repository Structure
```plaintext
TINYMOE/
├── .venv/                         # Python virtual environment
├── data/                          # Datasets and preprocessing artifacts
├── figures/                       # Figures used in the paper and documentation
├── results/
│   ├── arduino_models/            # Exported expert models for MCU deployment
│   ├── arduino_sketchs/           # Arduino sketches for embedded inference
│   ├── evaluation/                # Quantitative evaluation results
│   └── json_models/               # Serialized expert and router models
├── src/                           # Core source code
├── .gitignore
├── 01_dataset_generator.ipynb     # Dataset generation from PDF manuals
├── 02_training_router.ipynb       # Training of the semantic routing mechanism
├── 03_training_model.ipynb        # Training of TinyGPT domain experts
├── LICENSE
├── README.md
└── requirements.txt
````

## 📤 Installation

* **Python 3.9+** is recommended for reproducibility.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/conect2ai/acoustic_fault.git
cd acoustic_fault
python -m venv .venv
```

Download Dataset

```bash
aws configure set aws_access_key_id ...
aws configure set aws_secret_access_key ...
aws s3 cp s3://ieee-dataport/data/1383497/91533/Base_de_Dados_0.rar
```

Activate the environment:

```bash
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📄 Dataset

The acoustic [dataset](https://ieee-dataport.org/documents/acoustic-fault-dataset-internal-combustion-engine-under-controlled-conditions-ignition) was recorded from a **1.6 L gasoline internal combustion engine** test bench using a single omnidirectional microphone positioned at a fixed distance.

- **12 operating conditions**: 1 healthy baseline + 11 fault scenarios (cylinder misfires, V‑belt slippage, material loss, and combined faults)
- **2,148 labeled samples** (179 per class, perfectly balanced)
- Sampling rate: 16 kHz
- Segment length: 0.5 seconds (non‑overlapping windows)

Two spectral feature representations are extracted:

| Feature | Dimensionality | Description |
|---------|----------------|-------------|
| **MFCC** | 120 | 30 Mel‑Frequency Cepstral Coefficients + deltas + statistics (mean/std) |
| **MFE**  | 120 | 30 Mel‑Filterbank Energies (log‑compressed, no DCT) + deltas + statistics |

Data split: **80% training / 20% testing** (stratified, fixed random seed).

## 🧠 Proposed Approach

We evaluate five neural architectures for embedded acoustic fault classification under TinyML constraints (limited Flash, SRAM, and energy).

### Architectures

| Model | Key Primitive | Strengths | Weaknesses |
|-------|---------------|-----------|-------------|
| **FAN** | Trigonometric projections (sin/cos) + linear branch | High accuracy, very compact (34.5 KB) | Requires FPU for low latency |
| **CNN** | Local convolutions | Good accuracy, parameter sharing | Large memory footprint (227.5 KB) |
| **MLP** | Fully‑connected layers | Simple, predictable, low energy | Lower accuracy with MFE features |
| **RBFN** | Radial basis functions | Competitive accuracy | High inference cost (distance computations) |
| **KAN** | B‑spline edge functions | Expressive in theory | Prohibitive memory & latency |

### Feature Extraction Pipeline

1. Raw audio → resampling to 16 kHz → amplitude normalization
2. Frame‑level processing (25 ms frame, 10 ms stride, 512‑point FFT)
3. Mel filterbank (40 filters for MFCC, 30 for MFE)
4. Statistical aggregation over time (mean/std of static and delta coefficients)
5. Z‑score standardization (fit on training set only)

### Model Training & Selection

- Hyperparameter grid search (see Table 4 in paper)
- Training: 100 epochs, batch size 32, learning rate 1e‑3, weight decay 1e‑5
- Selection criteria: highest test accuracy, tie‑break by lower parameter count

### Python‑to‑C++ Conversion

- Static memory allocation (no dynamic allocation at inference)
- Dual‑buffer strategy for intermediate activations
- Weights placed in Flash (read‑only), activations in SRAM
- Architecture‑specific optimizations:
  - FAN: sine/cosine projections compiled to native math library calls
  - KAN: spline expressions converted to closed‑form algebraic equations
  - CNN/MLP: loop unrolling and matrix multiplication optimizations

### Microcontroller Deployment

Nine MCUs evaluated, covering **FPU‑equipped** and **software‑float** platforms:

| Platform | Core | FPU | RAM | Flash |
|----------|------|-----|-----|-------|
| Arduino Nano 33 BLE Sense | Cortex‑M4F | Yes | 256 KB | 1 MB |
| STM32F103C | Cortex‑M3 | No | 20 KB | 64 KB |
| ESP8266 | Tensilica L106 | No | 160 KB | 4 MB |
| Raspberry Pi Pico | RP2040 (M0+) | No | 264 KB | 2 MB |
| Raspberry Pi Pico W | RP2040 (M0+) | No | 264 KB | 2 MB |
| Raspberry Pi Pico 2 | RP2350 (M33) | Yes | 520 KB | 4 MB |
| Raspberry Pi Pico 2 W | RP2350 (M33) | Yes | 520 KB | 4 MB |
| ESP32 | Xtensa LX6 | Yes | 520 KB | 4 MB |
| Arduino Portenta H7 | Cortex‑M7+M4 | Yes | 1 MB | 16 MB |




### Research Questions

The study addresses the following research questions (**RQs**):

* **RQ1:** Does system-level MoE mitigate capacity saturation compared to a monolithic SLM?
* **RQ2:** How does semantic routing affect inference latency across heterogeneous MCUs?
* **RQ3:** What is the trade-off between Flash usage and real-time responsiveness?


## 📚 Jupyter Notebooks

-  [![Python](https://img.shields.io/badge/-Notebook-191A1B?style=flat-square&logo=jupyter)](https://github.com/conect2ai/acoustic_fault/blob/main/01_preprocessing.ipynb)  EDA and Preprocessing

-  [![Python](https://img.shields.io/badge/-Notebook-191A1B?style=flat-square&logo=jupyter)](https://github.com/conect2ai/acoustic_fault/blob/main/02_training_sweep.ipynb)  Training Grid

-  [![Python](https://img.shields.io/badge/-Notebook-191A1B?style=flat-square&logo=jupyter)](https://github.com/conect2ai/acoustic_fault/blob/main/03_counter_parameters.ipynb)  Counter C++ parameters

-  [![Python](https://img.shields.io/badge/-Notebook-191A1B?style=flat-square&logo=jupyter)](https://github.com/conect2ai/acoustic_fault/blob/main/04_result_visualization.ipynb)  Result visualization

## 📚 Arduino Code

-  [![Jupyter](https://img.shields.io/badge/Arduino-00878F?logo=arduino&logoColor=fff&style=plastic)](https://github.com/conect2ai/acoustic_fault/tree/main/results/arduino_code/CNN)  Convolutional Neural Networks (CNN)


-  [![Jupyter](https://img.shields.io/badge/Arduino-00878F?logo=arduino&logoColor=fff&style=plastic)](https://github.com/conect2ai/acoustic_fault/tree/main/results/arduino_code/FAN)  Fourier Analysis Networks (FAN)


-  [![Jupyter](https://img.shields.io/badge/Arduino-00878F?logo=arduino&logoColor=fff&style=plastic)](https://github.com/conect2ai/acoustic_fault/tree/main/results/arduino_code/MLP) Multi‑Layer Perceptrons (MLP)


-  [![Jupyter](https://img.shields.io/badge/Arduino-00878F?logo=arduino&logoColor=fff&style=plastic)](https://github.com/conect2ai/acoustic_fault/tree/main/results/arduino_code/KAN)  Kolmogorov‑Arnold Networks (KAN)


-  [![Jupyter](https://img.shields.io/badge/Arduino-00878F?logo=arduino&logoColor=fff&style=plastic)](https://github.com/conect2ai/acoustic_fault/tree/main/results/arduino_code/RBFN)  Radial Basis Function Networks (RBFN)


## 📊 Results Overview

### Classification Accuracy (Test Set)

| Architecture | MFCC Accuracy | MFE Accuracy |
|--------------|---------------|---------------|
| **FAN**      | **98.83%**    | 98.60%        |
| CNN          | 96.51%        | 98.25%        |
| MLP          | 96.04%        | 89.68%        |
| RBFN         | 97.30%        | 97.00%        |
| KAN          | 88.48%        | 88.13%        |

### Memory Footprint (Static Binary)

| Architecture | Size (KB) | Deployable on ESP8266? |
|--------------|-----------|------------------------|
| **FAN**      | **34.5**  | ✅ Yes                 |
| MLP          | 12.2      | ✅ Yes                 |
| CNN          | 227.5     | ❌ No                  |
| RBFN         | 281.2     | ❌ No                  |
| KAN          | 423–538   | ❌ No                  |

### Inference Latency & Energy (Selected Platforms)

| Platform (FPU) | Model | Latency (ms) | Energy (μJ) |
|----------------|-------|--------------|-------------|
| Portenta H7    | FAN   | 0.20         | 111         |
| ESP32          | FAN   | 0.67         | 142         |
| Pico 2         | FAN   | 2.04         | 166         |
| Pico 2         | MLP   | 0.14         | 10.7        |
| Pico (no FPU)  | FAN   | 20.7         | 2,608       |
| Pico (no FPU)  | MLP   | 4.24         | 566         |

### Key Findings

- ✅ **FAN achieves the best accuracy‑to‑resource trade‑off** on FPU‑equipped MCUs (98.83% accuracy, 34.5 KB footprint, sub‑ms latency).
- ⚡ **FPU is critical for FAN** – latency drops from 20.7 ms to 2.04 ms when moving from Pico (no FPU) to Pico 2 (FPU).
- 🧠 **MLP remains the most energy‑efficient** option on software‑float platforms (10.7 μJ per inference on Pico 2).
- ❌ **KAN is impractical** for TinyML – high memory (423+ KB) and latency (342 ms on Pico).
- 📊 **MFCC benefits MLP** (96% vs 89% with MFE); **FAN is robust** to feature choice (Δ < 0.3%).

## 🧠 Conclusion

This work provides a **hardware‑aware comparison** of five neural architectures for acoustic ignition fault detection under TinyML constraints.  

**Main contributions:**

1. **First systematic evaluation** of FAN, CNN, MLP, KAN, and RBFN for engine acoustic diagnosis across nine MCUs.
2. **Demonstration that FAN** achieves state‑of‑the‑art accuracy (98.83%) with a footprint of only 34.5 KB, making it deployable even on severely constrained devices.
3. **Quantification of FPU impact** – transcendental operations (sin/cos) dominate latency on software‑float platforms, while FPU reduces FAN latency by 10×.
4. **Pareto analysis** shows FAN dominates the high‑accuracy, low‑energy region on FPU‑equipped platforms; MLP is the safer choice for software‑float targets when energy is critical.

**Practical deployment rules:**

- **FPU available & high accuracy required** → **FAN + MFCC**
- **No FPU & energy/latency critical** → **MLP + MFCC**
- **No FPU but accuracy still important** → **FAN** (still fits memory, but expect higher latency)

**Limitations & future work:**  
The dataset was collected under controlled laboratory conditions. Real‑world validation (background noise, varying engine speeds, different vehicles) is needed. Future work will explore quantization for non‑FPU targets, multimodal sensing (audio + vibration), and on‑device adaptation.


## 📄 License

This package is licensed under the [MIT License](https://github.com/conect2ai/Conect2Py-Package/blob/main/LICENSE) - © 2026 Conect2ai.
