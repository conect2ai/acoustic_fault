# --- Bloco 3: Script de Inferência Explícita (infer.py) ---
# (Este bloco pode ser salvo como um arquivo separado)

import numpy as np
import torch # Usado APENAS para carregar o arquivo .pth
import os

# --- Funções de Ativação (NumPy) ---
def gelu(x):
    """Implementação NumPy da GELU [cite: 574]"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def get_activation_numpy(name):
    if name == 'gelu':
        return gelu
    # Adicione outras se necessário
    else:
        return lambda x: x # Identidade

# --- Funções de Camada (NumPy) ---

def linear_forward(x, W, B):
    """Calcula a inferência de uma camada linear: Wx + B"""
    return x @ W.T + B

def fan_layer_forward(x, W_p, W_p_bar, B_p_bar, activation_fn):
    """
    Inferência explícita da FANLayer (Eq. 9) 
    x: entrada (numpy array)
    W_p: pesos do ramo periódico (sem bias)
    W_p_bar: pesos do ramo MLP
    B_p_bar: bias do ramo MLP
    """
    # Ramo periódico
    x_p = x @ W_p.T
    periodic_cos = np.cos(x_p)
    periodic_sin = np.sin(x_p)
    
    # Ramo MLP
    mlp_part = activation_fn(x @ W_p_bar.T + B_p_bar)
    
    # Concatenação 
    return np.concatenate([periodic_cos, periodic_sin, mlp_part], axis=-1)

class ExplicitFAN:
    """
    Representa a "Função Simbólica" extraída.
    Carrega os pesos e executa a inferência usando apenas NumPy.
    """
    def __init__(self, weight_file, hparams):
        self.hparams = hparams
        self.weights = self._load_and_convert_weights(weight_file)
        self.activation_fn = get_activation_numpy(hparams['activation'])

    def _load_and_convert_weights(self, weight_file):
        """Carrega pesos do PyTorch e converte para NumPy."""
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Arquivo de pesos '{weight_file}' não encontrado.")
            
        # Carrega o state_dict do PyTorch
        state_dict = torch.load(weight_file)
        
        # Converte todos os tensores para arrays NumPy
        np_weights = {k: v.cpu().numpy() for k, v in state_dict.items()}
        print("Pesos carregados e convertidos para NumPy.")
        return np_weights

    def predict(self, x):
        """
        Executa a inferência (forward pass) manualmente.
        Isto É a sua função simbólica.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1) # Garante que a entrada seja 2D

        current_x = x
        n_layers = self.hparams['n_layers']
        
        # Itera pelas camadas ocultas (l < L) 
        for i in range(n_layers - 1):
            # Extrai os pesos desta camada pelo nome
            W_p = self.weights[f'layers.{i}.W_p.weight']
            W_p_bar = self.weights[f'layers.{i}.linear_p_bar.weight']
            B_p_bar = self.weights[f'layers.{i}.linear_p_bar.bias']
            
            current_x = fan_layer_forward(
                current_x, W_p, W_p_bar, B_p_bar, self.activation_fn
            )
            
        # Camada final (l = L) 
        final_layer_idx = n_layers - 1
        W_final = self.weights[f'layers.{final_layer_idx}.weight']
        B_final = self.weights[f'layers.{final_layer_idx}.bias']
        
        output = linear_forward(current_x, W_final, B_final)
        
        return output


    def export_to_arduino_universal(self, filename="fan_model.h", target="auto"):
        """
        Gera um único arquivo .h contendo todo o necessário para 
        executar a inferência no Arduino/compatíveis.
        
        Args:
            filename: Nome do arquivo de saída
            target: 'auto', 'avr', 'esp32', 'arm', 'generic'
        """
        
        # --- 1. Derivar dimensões (baseado nos HPARAMS) ---
        try:
            d_input = self.hparams['d_input']
            d_output = self.hparams['d_output']
            d_hidden = self.hparams['d_hidden']
            n_layers = self.hparams['n_layers']
            d_p_ratio = self.hparams['d_p_ratio']
            activation = self.hparams['activation']
        except KeyError:
            print("Erro: HPARAMS incompletos.")
            return

        # Calcula as dimensões das camadas
        layer_dims = []
        in_dim = d_input
        max_buffer_size = d_input
        
        for i in range(n_layers - 1):
            d_p = int(d_hidden * d_p_ratio)
            d_p_bar = d_hidden - (2 * d_p)
            layer_dims.append({
                'name': f'L{i}_fan',
                'type': 'fan',
                'in': in_dim,
                'out': d_hidden,
                'd_p': d_p,
                'd_p_bar': d_p_bar,
                'w_p_key': f'layers.{i}.W_p.weight',
                'w_p_bar_key': f'layers.{i}.linear_p_bar.weight',
                'b_p_bar_key': f'layers.{i}.linear_p_bar.bias',
            })
            in_dim = d_hidden
            max_buffer_size = max(max_buffer_size, d_hidden)
        
        # Camada final
        layer_dims.append({
            'name': f'L{n_layers - 1}_linear',
            'type': 'linear',
            'in': in_dim,
            'out': d_output,
            'w_key': f'layers.{n_layers - 1}.weight',
            'b_key': f'layers.{n_layers - 1}.bias',
        })
        max_buffer_size = max(max_buffer_size, d_output)

        # --- 2. Helper para formatar arrays C++ ---
        def format_array_cpp(arr, name):
            flat_arr = arr.flatten()
            rows, cols = (arr.shape[0], arr.shape[1]) if arr.ndim == 2 else (1, arr.shape[0])
            
            header = f"// {name} (Shape: {rows} x {cols})\n"
            header += f"const int {name}_rows = {rows};\n"
            header += f"const int {name}_cols = {cols};\n"
            header += f"const float {name}_data[]"
            
            # Para AVR, usa PROGMEM; para outros, apenas const
            if target == "avr":
                header += " PROGMEM"
            
            header += " = {\n  "
            
            vals = []
            for i, v in enumerate(flat_arr):
                vals.append(f"{v:.8e}f")
                if i < len(flat_arr) - 1:
                    vals.append(",")
                    if (i + 1) % 10 == 0:
                        vals.append("\n  ")
                    else:
                        vals.append(" ")
            
            return header + "".join(vals) + "\n};\n"

        # --- 3. Geração do código C++ universal ---
        cpp_code = f"""/*
    * Modelo FAN Gerado Automaticamente
    *
    * Arquitetura:
    * Entrada: {d_input}
    * Oculta: {d_hidden} (x{n_layers - 1})
    * Saída: {d_output}
    * Buffer Máx: {max_buffer_size}
    *
    * Uso:
    * #include "fan_model.h"
    *
    * float my_input[{d_input}] = {{ ... }};
    * float my_output[{d_output}];
    *
    * fan_predict(my_input, my_output);
    */

    #pragma once
    #include <math.h>
    #include <string.h>

    // Detecção automática de arquitetura
    #if defined(__AVR__)
    #include <avr/pgmspace.h>
    #define FAN_READ_FLOAT(p) pgm_read_float(p)
    #define FAN_PROGMEM PROGMEM
    #define FAN_BUFFER_STORAGE static
    #elif defined(ESP32)
    #include <rom/ets_sys.h>
    #define FAN_READ_FLOAT(p) (*(p))
    #define FAN_PROGMEM
    // CORREÇÃO: 'static' adicionado abaixo para evitar Stack Overflow no ESP32
    #define FAN_BUFFER_STORAGE static 
    #elif defined(__arm__) || defined(__ARM_ARCH)
    #define FAN_READ_FLOAT(p) (*(p))
    #define FAN_PROGMEM
    #define FAN_BUFFER_STORAGE static
    #else
    // Fallback para outras arquiteturas
    #define FAN_READ_FLOAT(p) (*(p))
    #define FAN_PROGMEM
    #define FAN_BUFFER_STORAGE static
    #endif

    // --- Dimensões Globais ---
    const int FAN_INPUT_SIZE = {d_input};
    const int FAN_OUTPUT_SIZE = {d_output};
    const int FAN_HIDDEN_SIZE = {d_hidden};
    const int FAN_MAX_BUFFER_SIZE = {max_buffer_size};

    // --- Funções de Ativação ---
    """

        # Adiciona função de ativação
        if activation == 'gelu':
            cpp_code += """
    static inline float fan_activation(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x3)));
    }
    """
        elif activation == 'relu':
            cpp_code += """
    static inline float fan_activation(float x) {
    return (x > 0.0f) ? x : 0.0f;
    }
    """
        else:
            cpp_code += "static inline float fan_activation(float x) { return x; }\n"

        # --- Funções de inferência universais ---
        cpp_code += """
    // Multiplicação Matriz-Vetor (W*x) + Bias (B)
    static void linear_forward_pgm(const float* x_in, float* x_out,
                                const float* W_pgm, const float* B_pgm,
                                int d_in, int d_out) {
    for (int i = 0; i < d_out; i++) {
        float sum = 0.0f;
        int w_offset = i * d_in;
        for (int j = 0; j < d_in; j++) {
        sum += FAN_READ_FLOAT(&W_pgm[w_offset + j]) * x_in[j];
        }
        x_out[i] = sum + FAN_READ_FLOAT(&B_pgm[i]);
    }
    }

    // Camada FAN
    static void fan_layer_forward_pgm(const float* x_in, float* x_out,
                                    const float* W_p_pgm, 
                                    const float* W_p_bar_pgm, 
                                    const float* B_p_bar_pgm,
                                    int d_in, int d_p, int d_p_bar) {
    
    // 1. Ramo Periódico
    for (int i = 0; i < d_p; i++) {
        float sum = 0.0f;
        int w_offset = i * d_in;
        for (int j = 0; j < d_in; j++) {
        sum += FAN_READ_FLOAT(&W_p_pgm[w_offset + j]) * x_in[j];
        }
        x_out[i] = cosf(sum);
        x_out[i + d_p] = sinf(sum);
    }

    // 2. Ramo MLP
    float* mlp_out_ptr = &x_out[2 * d_p];
    linear_forward_pgm(x_in, mlp_out_ptr, W_p_bar_pgm, B_p_bar_pgm, d_in, d_p_bar);

    // 3. Ativação
    for (int i = 0; i < d_p_bar; i++) {
        mlp_out_ptr[i] = fan_activation(mlp_out_ptr[i]);
    }
    }

    // --- Definição dos Pesos ---
    """

        # Adiciona os pesos
        weight_strings = []
        for layer in layer_dims:
            if layer['type'] == 'fan':
                W_p = self.weights[layer['w_p_key']]
                W_p_bar = self.weights[layer['w_p_bar_key']]
                B_p_bar = self.weights[layer['b_p_bar_key']]
                
                weight_strings.append(format_array_cpp(W_p, layer['name'] + "_Wp"))
                weight_strings.append(format_array_cpp(W_p_bar, layer['name'] + "_Wp_bar"))
                weight_strings.append(format_array_cpp(B_p_bar, layer['name'] + "_Bp_bar"))
            
            elif layer['type'] == 'linear':
                W = self.weights[layer['w_key']]
                B = self.weights[layer['b_key']]
                
                weight_strings.append(format_array_cpp(W, layer['name'] + "_W"))
                weight_strings.append(format_array_cpp(B, layer['name'] + "_B"))
        
        cpp_code += "\n".join(weight_strings)

        # --- Função de inferência principal (segura para qualquer MCU) ---
        cpp_code += f"""
    // --- Função de Inferência Principal ---
    void fan_predict(float* x_in, float* x_out) {{
    // Buffers de trabalho
    FAN_BUFFER_STORAGE float buffer1[FAN_MAX_BUFFER_SIZE];
    FAN_BUFFER_STORAGE float buffer2[FAN_MAX_BUFFER_SIZE];

    // Camada 0: L0_fan
    fan_layer_forward_pgm(
        x_in, buffer1,
        L0_fan_Wp_data, L0_fan_Wp_bar_data, L0_fan_Bp_bar_data,
        {layer_dims[0]['in']}, {layer_dims[0]['d_p']}, {layer_dims[0]['d_p_bar']}
    );
    """
        # Camadas intermediárias
        for i in range(1, len(layer_dims) - 1):
            cpp_code += f"""
    // Camada {i}: {layer_dims[i]['name']}
    fan_layer_forward_pgm(
        buffer1, buffer2,
        L{i}_fan_Wp_data, L{i}_fan_Wp_bar_data, L{i}_fan_Bp_bar_data,
        {layer_dims[i]['in']}, {layer_dims[i]['d_p']}, {layer_dims[i]['d_p_bar']}
    );
    
    // Troca os buffers
    memcpy(buffer1, buffer2, sizeof(float) * FAN_HIDDEN_SIZE);
    """
        # Última camada
        last = layer_dims[-1]
        cpp_code += f"""
    // Camada {len(layer_dims)-1}: {last['name']}
    linear_forward_pgm(
        buffer1, x_out,
        L{len(layer_dims)-1}_linear_W_data, L{len(layer_dims)-1}_linear_B_data,
        {last['in']}, {last['out']}
    );
    }}
    """
        
        # --- 4. Salvar o arquivo ---
        try:
            with open(filename, 'w') as f:
                f.write(cpp_code)
            print(f"✅ Arquivo gerado com sucesso: '{filename}'")
            print(f"   Target: {target}")
            print(f"   Tamanho do modelo: {sum(arr.size for arr in self.weights.values()) * 4 / 1024:.2f}KB")
        except Exception as e:
            print(f"❌ Erro ao salvar: {e}")

