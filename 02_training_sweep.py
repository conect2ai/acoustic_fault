import pandas as pd
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import warnings
import itertools
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



# --- Importações dos geradores de código ---
from fan_code_generation import ExplicitFAN
from kan_code_generation import generate_arduino_function_new

MFCC_CSV = "mfcc_dataset_onehot.csv"
MFE_CSV = "mfe_dataset_onehot.csv"

# --- Importações dos modelos (com fallback) ---
try:
    import cnn
except ImportError:
    print("Aviso: Módulo 'cnn' não encontrado.")

try:
    from fan import FAN
except ImportError:
    print("Aviso: Módulo 'fan' não encontrado.")

try:
    from kan import *
except ImportError:
    print("Aviso: Módulo 'kan' não encontrado.")

try:
    from tensorflores.models.multilayer_perceptron import MultilayerPerceptron
except ImportError:
    print("Aviso: Módulo 'tensorflores' não encontrado.")

try:
    from rbfn import AdvancedRBFNetwork
    from rbfn_code_generator import RBFArduinoCodeGenerator
except ImportError:
    print("Aviso: Módulo 'model.rbfn' não encontrado.")

warnings.filterwarnings("ignore")

# ============================
# FUNÇÕES AUXILIARES
# ============================

def get_total_cpp_header_size(folder_path: str) -> int:
    """Retorna o tamanho total (bytes) dos arquivos .cpp e .h em folder_path."""
    total = 0
    for fname in os.listdir(folder_path):
        if fname.endswith(('.cpp', '.h')):
            total += os.path.getsize(os.path.join(folder_path, fname))
    return total

def load_data(feature_type='mfcc'):
    """
    Carrega dados dos CSVs MFCC e MFE já em formato numérico
    com labels one-hot.
    """

    if feature_type == 'mfcc':
        df = pd.read_csv(MFCC_CSV)

    elif feature_type == 'mfe':
        df = pd.read_csv(MFE_CSV)

    elif feature_type == 'concat':
        df_mfcc = pd.read_csv(MFCC_CSV)
        df_mfe = pd.read_csv(MFE_CSV)

        # assume que labels são iguais
        label_cols = df_mfcc.columns[-10:]

        X_mfcc = df_mfcc.iloc[:, :-10].values
        X_mfe = df_mfe.iloc[:, :-10].values

        X = np.concatenate([X_mfcc, X_mfe], axis=1)
        y = df_mfcc[label_cols].values.astype(np.float32)

        return X.astype(np.float32), y, list(label_cols)

    else:
        raise ValueError("feature_type must be: mfcc, mfe or concat")

    # Detect label columns automatically (one-hot)
    label_cols = [c for c in df.columns if df[c].isin([0,1]).all()]

    X = df.drop(columns=label_cols).values.astype(np.float32)
    y = df[label_cols].values.astype(np.float32)

    return X, y, label_cols

# ============================
# FUNÇÕES DE TREINAMENTO (SEM WANDB)
# ============================

def train_cnn(X_train, y_train, X_test, y_test, config, run_id):
    """
    Trains a CNN using PyTorch with architecture defined by config.
    
    Parameters:
        X_train, y_train: training data and labels (one-hot encoded)
        X_test, y_test: test data and labels (one-hot encoded)
        config: dictionary with keys:
            cnn_conv_layers, cnn_filters, cnn_kernel, cnn_dense_layers,
            cnn_dense_units, lr, epochs, batch_size
        run_id: identifier for saving outputs
    Returns:
        (y_true_test, y_pred_test, y_true_train, y_pred_train,
         avg_inf_time, train_time, size_file)
    """
    # Extract config parameters
    conv_layers = int(config.get('cnn_conv_layers', 2))
    filters = int(config.get('cnn_filters', 16))
    kernel_size = int(config.get('cnn_kernel', 3))
    dense_layers = int(config.get('cnn_dense_layers', 1))
    dense_units = int(config.get('cnn_dense_units', 32))
    lr = float(config.get('lr', 0.001))
    epochs = int(config.get('epochs', 10))
    batch_size = int(config.get('batch_size', 32))
    
    # --- Input reshaping (same as original) ---
    input_dim = X_train.shape[1]
    h = int(np.sqrt(input_dim))
    w = input_dim // h
    if h * w < input_dim:
        w += 1
    pad = h * w - input_dim
    if pad > 0:
        X_train = np.pad(X_train, ((0,0),(0,pad)), mode='constant')
        X_test  = np.pad(X_test,  ((0,0),(0,pad)), mode='constant')
    
    # Reshape to (N, 1, H, W) - add channel dimension
    X_train_reshaped = X_train.reshape(-1, 1, h, w).astype(np.float32)
    X_test_reshaped = X_test.reshape(-1, 1, h, w).astype(np.float32)
    
    # Convert labels from one-hot to class indices
    y_train_idx = np.argmax(y_train, axis=1).astype(np.int64)
    y_test_idx = np.argmax(y_test, axis=1).astype(np.int64)
    
    # --- Build dynamic PyTorch model as a Sequential ---
    layers = []
    in_channels = 1
    current_h, current_w = h, w
    
    # Convolutional blocks
    for _ in range(conv_layers):
        if current_h < kernel_size or current_w < kernel_size:
            break
        layers.append(nn.Conv2d(in_channels, filters, kernel_size=kernel_size))
        layers.append(nn.ReLU())
        if current_h >= 2 and current_w >= 2:
            layers.append(nn.MaxPool2d(2))
            current_h //= 2
            current_w //= 2
        # Update dimensions after convolution (no padding)
        current_h = current_h - kernel_size + 1
        current_w = current_w - kernel_size + 1
        in_channels = filters
    
    # Flatten
    layers.append(nn.Flatten())
    # Compute flattened size
    with torch.no_grad():
        dummy = torch.zeros(1, 1, h, w)
        for layer in layers:
            dummy = layer(dummy)
        flattened_size = dummy.numel()
    
    # Dense layers
    in_features = flattened_size
    for _ in range(dense_layers):
        layers.append(nn.Linear(in_features, dense_units))
        layers.append(nn.ReLU())
        in_features = dense_units
    # Output layer (no activation, CrossEntropyLoss handles softmax)
    layers.append(nn.Linear(in_features, y_train.shape[1]))
    
    # Wrap the Sequential in a class that exposes .layers for export compatibility
    class WrappedSequential(nn.Module):
        def __init__(self, sequential):
            super().__init__()
            self.layers = sequential  # this attribute is expected by export_to_json
        
        def forward(self, x):
            return self.layers(x)
    
    model = WrappedSequential(nn.Sequential(*layers))
    
    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # --- Data preparation ---
    train_dataset = TensorDataset(torch.from_numpy(X_train_reshaped),
                                  torch.from_numpy(y_train_idx))
    test_dataset = TensorDataset(torch.from_numpy(X_test_reshaped),
                                 torch.from_numpy(y_test_idx))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --- Loss and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # --- Training loop ---
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
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
        
        train_acc = correct / total
        print(f"Epoch {epoch}/{epochs} - Loss: {running_loss/total:.4f}, Acc: {train_acc:.4f}")
    
    train_time = time.time() - start_time
    
    # --- Evaluation on test set (per-sample timing) ---
    model.eval()
    y_pred_test = []
    inf_times = []
    with torch.no_grad():
        for x in X_test_reshaped:
            t0 = time.time()
            x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)  # add batch dim
            output = model(x_tensor)
            pred = torch.argmax(output, dim=1).cpu().item()
            inf_times.append(time.time() - t0)
            y_pred_test.append(pred)
    avg_inf_time = np.mean(inf_times)
    
    # Predictions on train set
    y_pred_train = []
    with torch.no_grad():
        for x in X_train_reshaped:
            x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
            output = model(x_tensor)
            pred = torch.argmax(output, dim=1).cpu().item()
            y_pred_train.append(pred)
    
    y_true_test = y_test_idx.tolist()
    y_true_train = y_train_idx.tolist()
    
    # --- Export to JSON and generate C++ code ---
    size_file = 0
    try:
        out_dir = f"outputs/run_{run_id}/cnn_cpp"
        os.makedirs(out_dir, exist_ok=True)
        json_path = f"{out_dir}/model.json"
        # Move model to CPU for export
        model.to('cpu')
        # Assumes export_to_json and generate_arduino_code are defined elsewhere
        cnn.export_to_json(model, json_path, input_shape=(1, h, w))
        cnn.generate_arduino_code(json_path, out_dir,
                              input_height=h, input_width=w)
        # Compute total size of generated files
        for fname in ['neural_network.h', 'neural_network.cpp']:
            path = os.path.join(out_dir, fname)
            if os.path.exists(path):
                size_file += os.path.getsize(path)
    except Exception as e:
        print(f"Error in C++ generation: {e}")
    
    #print('x_tensor: ', X_train_reshaped[0].reshape(1,-1).tolist())
    #print('y_true_train: ', y_true_train[0])
    #print('y_pred_train: ', y_pred_train[0])
    return (y_true_test, y_pred_test, y_true_train, y_pred_train,
            avg_inf_time, train_time, size_file)


def train_fan(X_train, y_train, X_test, y_test, config, run_id):
    # 1. Extração de Hiperparâmetros (agora incluindo o batch_size)
    fan_hidden = int(config.get('fan_hidden', 64))
    fan_layers = int(config.get('fan_layers', 3))
    d_p_ratio = float(config.get('d_p_ratio', 0.25))
    fan_activation = config.get('fan_activation', 'gelu')
    lr = float(config.get('lr', 0.001)) 
    epochs = int(config.get('epochs'))
    batch_size = int(config.get('batch_size', 32))
    optimizer_name = config.get('optimizer', 'adam').lower()
    weight_decay = float(config.get('weight_decay', 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Preparação de Dados com DataLoader para Mini-batches
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    # CrossEntropyLoss espera os labels como índices inteiros (LongTensor)
    y_train_labels = torch.argmax(y_train_t, dim=1).long() 
    
    train_dataset = TensorDataset(X_train_t, y_train_labels)
    # Shuffle=True é essencial para o modelo não decorar a ordem dos dados
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 3. Inicialização do Modelo (Atenção: a classe FAN não deve ter Softmax na saída)
    model = FAN(
        d_input=X_train.shape[1],
        d_output=y_train.shape[1],
        d_hidden=fan_hidden,
        n_layers=fan_layers,
        d_p_ratio=d_p_ratio,
        activation=fan_activation
    ).to(device)

    # 4. Configuração do Otimizador
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # SGD com momentum geralmente tem um desempenho melhor que SGD puro
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # CrossEntropyLoss calcula o LogSoftmax + NLLLoss internamente
    criterion = torch.nn.CrossEntropyLoss()

    # 5. Loop de Treinamento
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X) # Gera os logits puros
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Calcula a loss média da época considerando todos os batches
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        if (epoch+1) % 10 == 0:
            print(f"FAN Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
            
    train_time = time.time() - start_time

    # 6. Avaliação e Inferência
    model.eval()
    y_pred_test, inf_times = [], []
    
    # Medindo o tempo de inferência amostra por amostra (simulando ambiente real/embarcado)
    with torch.no_grad():
        for i in range(len(X_test)):
            t0 = time.time()
            out = model(X_test_t[i:i+1])
            inf_times.append(time.time() - t0)
            # argmax funciona perfeitamente com logits puros
            y_pred_test.append(out.argmax(dim=1).cpu().item())
            
    avg_inf_time = np.mean(inf_times)
    y_true_test = np.argmax(y_test, axis=1)

    # Predição no conjunto de treino (usando batch inteiro para ser mais rápido)
    with torch.no_grad():
        X_train_full = X_train_t.to(device)
        out_train = model(X_train_full)
        y_pred_train = out_train.argmax(dim=1).cpu().numpy().tolist()
    y_true_train = np.argmax(y_train, axis=1)

    # 7. Salvar Modelo e Exportar Código C++
    out_dir = f"outputs/run_{run_id}/fan_model"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/weights.pth"
    torch.save(model.state_dict(), model_path)

    hparams = {
        'd_input': X_train.shape[1],
        'd_output': y_train.shape[1],
        'd_hidden': fan_hidden,
        'n_layers': fan_layers,
        'd_p': d_p_ratio,          # <--- ESTA É A CHAVE QUE A CLASSE EXPLICITFAN EXIGE
        'd_p_ratio': d_p_ratio,    # Mantemos esta por garantia para outras partes do código
        'activation': fan_activation,
        'lr': lr,
        'epochs': epochs,
        'weight_file': model_path
    }
    
    # Exportação para o microcontrolador
    explicit_gen = ExplicitFAN(model_path, hparams)
    explicit_gen.export_to_arduino_universal(
        f"{out_dir}/fan_model.h", target="esp32"
    )
    size_file = os.path.getsize(model_path)

    return (y_true_test, y_pred_test, y_true_train, y_pred_train,
            avg_inf_time, train_time, size_file)


def train_kan(X_train, y_train, X_test, y_test, config, run_id):
    # --- Configurações de Hiperparâmetros ---
    batch_size = int(config.get('batch_size', 32))
    kan_hidden = int(config.get('kan_hidden', 20))
    kan_grid = int(config.get('kan_grid', 5))
    kan_k = int(config.get('kan_k', 3))
    kan_lambda = float(config.get('kan_lambda', 0.01))
    lr = float(config.get('lr', 0.001))
    epochs = int(config.get('epochs'))
    optimizer_name = config.get('optimizer', 'adam')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.get_default_dtype()

    # Rótulos para métricas (índices)
    y_train_idx = np.argmax(y_train, axis=1)
    y_test_idx = np.argmax(y_test, axis=1)

    # Dataset no formato esperado pela KAN (one-hot para MSE)
    dataset = {
        'train_input': torch.from_numpy(X_train).type(dtype).to(device),
        'test_input':  torch.from_numpy(X_test).type(dtype).to(device),
        'train_label': torch.from_numpy(y_train).type(dtype).to(device),
        'test_label':  torch.from_numpy(y_test).type(dtype).to(device),
    }

    # --- 1. Criação do modelo com arquitetura correta ---
    model = KAN(
        width=[X_train.shape[1], kan_hidden, y_train.shape[1]],
        grid=kan_grid,
        k=kan_k,
        seed=42
    ).to(device)

    # --- 2. Treinamento com número adequado de passos ---
    steps_per_epoch = 1
    total_steps = epochs * steps_per_epoch

    start_time = time.time()
    model.fit(
        dataset,
        opt=optimizer_name,
        lr=lr,
        steps=total_steps,
        batch=batch_size,
        lamb=kan_lambda,
        update_grid=True,
        singularity_avoiding=True
    )
    train_time = time.time() - start_time

    # --- 3. Avaliação do modelo original (antes da extração simbólica) ---
    model.eval()
    with torch.no_grad():
        train_logits = model(dataset['train_input'])
        test_logits  = model(dataset['test_input'])

    y_pred_train = torch.argmax(train_logits, dim=1).cpu().numpy()
    y_pred_test  = torch.argmax(test_logits, dim=1).cpu().numpy()

    # Tempo de inferência médio (Isso sobrescreve o cache do modelo com apenas 1 amostra)
    inf_times = []
    for i in range(len(X_test)):
        t0 = time.time()
        _ = model(dataset['test_input'][i:i+1])
        inf_times.append(time.time() - t0)
    avg_inf_time = np.mean(inf_times)

    # --- 4. Extração Simbólica ---
    
    # === INÍCIO DA CORREÇÃO ===
    # Retornamos ao modo de treino e passamos o dataset completo novamente
    # para preencher o cache interno com uma distribuição rica de ativações.
    model.train()
    with torch.no_grad():
        _ = model(dataset['train_input'])
    # === FIM DA CORREÇÃO ===

    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','sin','tan','abs']
    model.auto_symbolic(lib=lib)          # substitui as funções internas

    formula = model.symbolic_formula()
    symbol_list = []
    for symbol in formula[0]:              # mantém a lógica original (apenas primeira saída)
        symbol_list.append(str(symbol))

    # --- 5. Geração do código C++ ---
    out_dir = f"outputs/run_{run_id}/kan_cpp"
    os.makedirs(out_dir, exist_ok=True)
    header_path = f"{out_dir}/kan_model.h"

    generate_arduino_function_new(symbol_list, X_train.shape[1], header_path)

    size_file = os.path.getsize(header_path) if os.path.exists(header_path) else 0

    return (y_test_idx, y_pred_test, y_train_idx, y_pred_train,
            avg_inf_time, train_time, size_file)


def train_mlp(X_train, y_train, X_test, y_test, config, run_id):
    mlp_scenario = config.get('mlp_scenario', 0)
    mlp_init_weight = config.get('mlp_init_weight', 'RandomNormal')
    lr = config.get('lr', 0.001)
    epochs = config.get('epochs')
    batch_size = config.get('batch_size', 32)

    hidden_sizes = [[16], [32, 16], [64, 32, 16], [128, 64, 32, 16]]
    activations = [
        ['relu', 'softmax'],
        ['relu', 'relu', 'softmax'],
        ['relu', 'relu', 'relu', 'softmax'],
        ['relu', 'relu', 'relu', 'relu', 'softmax']
    ]
    scenario = mlp_scenario

    nn = MultilayerPerceptron(
        input_size=X_train.shape[1],
        output_size=y_train.shape[1],
        hidden_layer_sizes=hidden_sizes[scenario],
        activation_functions=activations[scenario],
        weight_bias_init=mlp_init_weight,
        training_with_quantization=False
    )

    start_time = time.time()
    nn.train(X=X_train, y=y_train, epochs=epochs,
             learning_rate=lr, batch_size=batch_size)
    train_time = time.time() - start_time

    t0 = time.time()
    preds_test = nn.predict(X_test)
    y_pred_test = np.argmax(preds_test, axis=1)
    avg_inf_time = (time.time() - t0) / len(X_test)
    y_true_test = np.argmax(y_test, axis=1)

    preds_train = nn.predict(X_train)
    y_pred_train = np.argmax(preds_train, axis=1)
    y_true_train = np.argmax(y_train, axis=1)

    out_dir = f"outputs/run_{run_id}/mlp_cpp"
    os.makedirs(out_dir, exist_ok=True)
    nn.save_model_as_cpp(os.path.join(out_dir, "MLP_model"))
    size_file = get_total_cpp_header_size(out_dir)

    return (y_true_test, y_pred_test, y_true_train, y_pred_train,
            avg_inf_time, train_time, size_file)


def train_rbfn(X_train, y_train, X_test, y_test, config, run_id):
    rbf_units = config.get('rbf_units', 50)
    rbf_type = config.get('rbf_type', 'gaussian')
    width_method = config.get('rbf_width_method', 'p_nearest')
    if width_method == 'fixed':
        width_val = config.get('rbf_fixed_width', 1.0)
    else:
        width_val = config.get('rbf_p_neighbors', 2)
    rbf_reg = config.get('rbf_regularization', 0.0)
    epochs = config.get('epochs')

    model = AdvancedRBFNetwork(
        hidden_layers=[{
            'type': 'rbf',
            'units': rbf_units,
            'center_init': 'kmeans',
            'width_calc': width_method
        }],
        rbf_types=rbf_type,
        training_method='direct',
        max_iter=epochs,
        regularization=rbf_reg,
        p_neighbors=width_val if width_method == 'p_nearest' else 2,
        width_value=width_val if width_method == 'fixed' else 1.0,
        verbose=False
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    preds_test = model.predict(X_test)
    y_pred_test = np.argmax(preds_test, axis=1)
    y_true_test = np.argmax(y_test, axis=1)

    preds_train = model.predict(X_train)
    y_pred_train = np.argmax(preds_train, axis=1)
    y_true_train = np.argmax(y_train, axis=1)

    inf_times = []
    for x in X_test:
        t0 = time.time()
        _ = model.predict(x.reshape(1, -1))
        inf_times.append(time.time() - t0)
    avg_inf_time = np.mean(inf_times)

    out_dir = f"outputs/run_{run_id}/rbfn_cpp"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "rbf_model.json")
    model.save_json(json_path)

    try:
        gen = RBFArduinoCodeGenerator(json_path)
        gen.generate(out_dir, target_device="esp32")
    except Exception as e:
        print(f"Erro na geração C++ do RBFN: {e}")

    size_file = get_total_cpp_header_size(out_dir)

    return (y_true_test, y_pred_test, y_true_train, y_pred_train,
            avg_inf_time, train_time, size_file)


# ============================
# FUNÇÃO PRINCIPAL DE EXECUÇÃO (SEM WANDB)
# ============================

def run_experiment(config, run_id):
    """Executa uma combinação de hiperparâmetros e retorna um dicionário com os resultados."""
    model_name = config.get('model_name', 'CNN')
    feature_type = config.get('feature_type', 'mfcc')

    # Carregar dados
    X, y, target_names = load_data(feature_type=feature_type)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_funcs = {
        'CNN': train_cnn,
        'FAN': train_fan,
        'MLP': train_mlp,
        'KAN': train_kan,
        'RBFN': train_rbfn
    }

    if model_name not in model_funcs:
        raise ValueError(f"Modelo {model_name} desconhecido")

    result = model_funcs[model_name](X_train, y_train, X_test, y_test,
                                      config, run_id)

    (y_true_test, y_pred_test, y_true_train, y_pred_train,
     avg_inf_time, train_time, size_file) = result

    # Métricas
    acc_test = accuracy_score(y_true_test, y_pred_test)
    f1_test = f1_score(y_true_test, y_pred_test, average='weighted')
    prec_test = precision_score(y_true_test, y_pred_test, average='weighted')
    rec_test = recall_score(y_true_test, y_pred_test, average='weighted')

    acc_train = accuracy_score(y_true_train, y_pred_train)
    f1_train = f1_score(y_true_train, y_pred_train, average='weighted')
    prec_train = precision_score(y_true_train, y_pred_train, average='weighted')
    rec_train = recall_score(y_true_train, y_pred_train, average='weighted')

    # Monta dicionário de resultados (inclui todos os hiperparâmetros)
    row = {
        'run_id': run_id,
        'model_name': model_name,
        'feature_type': feature_type,
        'epochs': config.get('epochs'),
        'lr': config.get('lr'),
        'batch_size': config.get('batch_size'),
        'optimizer': config.get('optimizer'),
        'weight_decay': config.get('weight_decay'),
        # Hiperparâmetros específicos (serão preenchidos com None se não existirem)
        'cnn_conv_layers': config.get('cnn_conv_layers'),
        'cnn_filters': config.get('cnn_filters'),
        'cnn_kernel': config.get('cnn_kernel'),
        'cnn_dense_layers': config.get('cnn_dense_layers'),
        'cnn_dense_units': config.get('cnn_dense_units'),
        'cnn_dropout': config.get('cnn_dropout'),
        'fan_hidden': config.get('fan_hidden'),
        'fan_layers': config.get('fan_layers'),
        'd_p_ratio': config.get('d_p_ratio'),
        'fan_activation': config.get('fan_activation'),
        'mlp_scenario': config.get('mlp_scenario'),
        'mlp_init_weight': config.get('mlp_init_weight'),
        'mlp_dropout': config.get('mlp_dropout'),
        'kan_hidden': config.get('kan_hidden'),
        'kan_grid': config.get('kan_grid'),
        'kan_k': config.get('kan_k'),
        'kan_lambda': config.get('kan_lambda'),
        'rbf_units': config.get('rbf_units'),
        'rbf_type': config.get('rbf_type'),
        'rbf_width_method': config.get('rbf_width_method'),
        'rbf_p_neighbors': config.get('rbf_p_neighbors'),
        'rbf_fixed_width': config.get('rbf_fixed_width'),
        'rbf_regularization': config.get('rbf_regularization'),
        # Métricas
        'accuracy_test': acc_test,
        'f1_score_test': f1_test,
        'precision_test': prec_test,
        'recall_test': rec_test,
        'accuracy_train': acc_train,
        'f1_score_train': f1_train,
        'precision_train': prec_train,
        'recall_train': rec_train,
        'inference_time_sec': avg_inf_time,
        'training_time_sec': train_time,
        'cpp_code_size_bytes': size_file,
    }

    print(f"[run {run_id}] Modelo: {model_name} | Acc: {acc_test:.4f} | F1: {f1_test:.4f}")
    return row


def load_sweep_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_grid(config):
    """Gera todas as combinações de parâmetros a partir da seção 'parameters' do YAML."""
    param_grid = {}
    for key, value in config['parameters'].items():
        if 'values' in value:
            param_grid[key] = value['values']
    keys = param_grid.keys()
    values = param_grid.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


if __name__ == "__main__":
    # Carrega configuração do sweep
    sweep_config = load_sweep_config('sweep_config.yaml')

    # Cria diretório base para saídas
    os.makedirs('outputs', exist_ok=True)

    # --- INÍCIO DO MECANISMO DE CHECKPOINT ---
    csv_path = 'resultados_sweep.csv'
    completed_run_ids = set()

    # Se o arquivo CSV já existe, carrega os run_ids já processados
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            # Garantir que a coluna run_id existe
            if 'run_id' in df_existing.columns:
                completed_run_ids = set(df_existing['run_id'].unique())
                print(f"Checkpoint carregado: {len(completed_run_ids)} execuções já concluídas.")
            else:
                print("Aviso: arquivo CSV existente não contém coluna 'run_id'. Ignorando checkpoint.")
        except Exception as e:
            print(f"Erro ao ler o arquivo de checkpoint: {e}. Prosseguindo sem checkpoint.")
    # --- FIM DO MECANISMO DE CHECKPOINT ---

    # Itera sobre todas as combinações do grid
    for run_id, params in enumerate(generate_grid(sweep_config)):
        # Pula se já foi processado
        if run_id in completed_run_ids:
            print(f"Run {run_id} já concluída. Pulando...")
            continue

        print(f"\n=== Executando run {run_id} com parâmetros: {params} ===")
        try:
            row = run_experiment(params, run_id)

            # --- Salvamento incremental ---
            # Converte a linha em DataFrame e anexa ao CSV
            df_row = pd.DataFrame([row])
            # Se o arquivo não existir, escreve cabeçalho; caso contrário, apenas anexa
            header = not os.path.exists(csv_path)
            df_row.to_csv(csv_path, mode='a', header=header, index=False)
            print(f"Resultado da run {run_id} salvo em {csv_path}")

            # Opcional: adiciona à lista de concluídos em memória para evitar reprocessamento
            # (já que estamos pulando pelo checkpoint, mas pode ser útil se houver mais de um processo)
            completed_run_ids.add(run_id)

        except Exception as e:
            print(f"Erro na execução {run_id}: {e}")
            # Não salva nada no CSV, então a run será tentada novamente na próxima reinicialização
            continue

    print(f"\nProcesso concluído. Resultados finais em '{csv_path}'.")




"""
program: train_sweep.py
method: grid                # pode ser alterado para 'bayes' ou 'random'
metric:
  name: accuracy_test
  goal: maximize

parameters:
  # ========== HIPERPARÂMETROS GLOBAIS ==========
  model_name:
    values: ['KAN']
  feature_type:
    values: ['mfcc', 'mfe']
  epochs:
    values: [100]
  lr:
    values: [0.001]
  batch_size:
    values: [32]
  optimizer:
    values: ["Adam", "LBFGS"] # KAN: Adam, LBFGS
  weight_decay:
    values: [1e-5]

  # ========== KAN ==========
  kan_hidden:
    values: [10, 20, 30, 50]
  kan_grid:
    values: [3, 5, 7]
  kan_k:
    values: [3, 5]                 # ordem da spline
  kan_lambda:
    values: [0.01, 0.1, 1.0]       # regularização
 
"""