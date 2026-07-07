import torch
import torch.nn as nn

class StencilMLP_Global(nn.Module):
    def __init__(self, max_n=60, features_por_no=3, hidden_dim=256):
        super(StencilMLP_Global, self).__init__()
        
        tamanho_entrada = max_n * features_por_no
        
        self.net = nn.Sequential(
            # Camada 1: Entrada -> Oculta 1
            nn.Linear(tamanho_entrada, hidden_dim),
            nn.LeakyReLU(0.1),
            
            # Camada 2: Oculta 1 -> Oculta 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            
            # Camada 3: Oculta 2 -> Oculta 3
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),

            # Camada 4: Oculta 3 -> Oculta 4
            nn.Linear(hidden_dim, hidden_dim // 2), # 256 -> 128
            nn.LeakyReLU(0.1),
            
            # Camada 5: Oculta 4 -> Oculta 5
            nn.Linear(hidden_dim // 2, hidden_dim // 4), # 128 -> 64
            nn.LeakyReLU(0.1),
            
            # Camada 6: Oculta 5 -> Output
            nn.Linear(hidden_dim // 4, max_n) # 64 -> 60
        )

    def forward(self, x):
        # x já entra achatado (flattened) no formato [Batch, MAX_N * 4]
        pesos_preditos = self.net(x)
        return pesos_preditos
    


def masked_loss(pred_weights, true_weights, mask):
    #Utilizando erro relativo para sanar o problema de distâncias muito baixas com erros muito altos
    erro_relativo = (pred_weights - true_weights)**2
    erro_mascarado = erro_relativo * mask
    
    # Média apenas dos pontos válidos
    mse_loss = torch.sum(erro_mascarado) / torch.sum(mask)
    return mse_loss

def masked_loss_relative(pred_weights, true_weights, mask):
    #Utilizando erro relativo para sanar o problema de distâncias muito baixas com erros muito altos
    erro_relativo = (pred_weights - true_weights)**2 / abs(true_weights)
    erro_mascarado = erro_relativo * mask
    
    # Média apenas dos pontos válidos
    mse_loss = torch.sum(erro_mascarado) / torch.sum(mask)
    return mse_loss