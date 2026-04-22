import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# --- Bloco 1: Definição das Camadas FAN (Baseado no Artigo) ---

def get_activation(name):
    """Retorna a função de ativação do PyTorch pelo nome."""
    if name == 'gelu':
        return nn.GELU()  # Usado nos experimentos do artigo [cite: 574]
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'silu':
        return nn.SiLU()
    else:
        return nn.Identity()

class FANLayer(nn.Module):
    """
    Implementação da camada FAN, baseada na Equação 9 do artigo.
    φ(x) = [cos(W_p*x) || sin(W_p*x) || σ(B_p_bar + W_p_bar*x)]
    
    """
    def __init__(self, d_input, d_output, d_p_ratio=0.25, activation_fn=nn.GELU()):
        super().__init__()
        
        # d_p é a dimensão para os componentes periódicos. O artigo usa 1/4 da dimensão [cite: 90, 149]
        self.d_p = int(d_output * d_p_ratio)
        
        # d_p_bar é a dimensão para o componente MLP
        self.d_p_bar = d_output - (2 * self.d_p)
        
        if self.d_p_bar < 0:
            raise ValueError(f"d_output ({d_output}) é muito pequeno para d_p_ratio ({d_p_ratio}).")

        # Ramo periódico (W_p): sem bias
        self.W_p = nn.Linear(d_input, self.d_p, bias=False)
        
        # Ramo MLP (W_p_bar, B_p_bar)
        self.linear_p_bar = nn.Linear(d_input, self.d_p_bar, bias=True)
        
        self.activation = activation_fn

    def forward(self, x):
        # Parte periódica
        x_p = self.W_p(x)
        periodic_cos = torch.cos(x_p)
        periodic_sin = torch.sin(x_p)
        
        # Parte MLP
        mlp_part = self.activation(self.linear_p_bar(x))
        
        # Concatenação 
        return torch.cat([periodic_cos, periodic_sin, mlp_part], dim=-1)

class FAN(nn.Module):
    """
    Modelo FAN completo, empilhando camadas FAN e uma camada linear final.
    Baseado nas Equações 10 e 11. [cite: 140, 141]
    """
    def __init__(self, d_input, d_output, d_hidden, n_layers, d_p_ratio=0.25, activation='gelu'):
        super().__init__()
        
        self.layers = nn.ModuleList()
        act_fn = get_activation(activation)
        
        in_dim = d_input
        
        # Camadas ocultas (l < L) 
        for _ in range(n_layers - 1):
            self.layers.append(FANLayer(
                d_input=in_dim,
                d_output=d_hidden,
                d_p_ratio=d_p_ratio,
                activation_fn=act_fn
            ))
            in_dim = d_hidden
            
        # Camada final (l = L) é uma camada linear padrão 
        self.layers.append(nn.Linear(in_dim, d_output))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x