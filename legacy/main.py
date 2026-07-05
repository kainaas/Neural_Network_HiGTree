import torch
import torch.nn as nn
from data_treat import load_data
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from torch.utils.data import random_split
from torch_geometric.utils import scatter
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================
# 1. ARQUITETURA DA GNN (Graph Neural Network)
# ==========================================
class StencilGNN(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=32, heads_layer1=4, heads_layer2=4):
        super(StencilGNN, self).__init__()
        # in_channels = 4 (dx_norm, dy_norm, razao_x, razao_y) que combinamos antes!
        
        # 1ª Camada: Concatena as cabeças. 
        # Saída real desta camada será: hidden_channels * heads_layer1
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads_layer1, concat=True)
        
        # 2ª Camada: Precisa receber o tamanho exato que a conv1 cuspiu!
        dimensao_entrada_conv2 = hidden_channels * heads_layer1
        
        # Nesta camada, usamos concat=False. 
        # Isso tira a MÉDIA das cabeças em vez de concatenar, garantindo que a 
        # saída volte a ter o tamanho exato de 'hidden_channels'.
        self.conv2 = GATConv(dimensao_entrada_conv2, hidden_channels, heads=heads_layer2, concat=False)
        
        # Camada final linear: Como a conv2 tirou a média, o tamanho aqui é só hidden_channels
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # Passa pela primeira camada com ativação ELU
        h = self.conv1(x, edge_index)
        h = torch.nn.functional.elu(h)
        
        # Passa pela segunda camada
        h = self.conv2(h, edge_index)
        h = torch.nn.functional.elu(h)
        
        # Camada de saída que prevê 1 peso por nó vizinho
        weights = self.out(h).squeeze(-1) 
        return weights

# ==========================================
# 2. FUNÇÃO DE PERDA COM RESTRIÇÃO FÍSICA
# ==========================================
def physics_informed_loss(pred_weights, true_weights, batch, target_sum=0.0, lambda_reg=0.1):
    """
    target_sum = 0.0 (Para Derivadas/Laplaciano)
    target_sum = 1.0 (Para Interpolação de valores)
    lambda_reg = Força da penalidade física (ajuste fino necessário)
    """
    # 1. Erro Matemático (MSE padrão contra os pesos do WLS exato)
    mse_loss = torch.mean((pred_weights - true_weights)**2)
    
    # 2. Restrição Física (A soma dos pesos de um stencil deve ser 'target_sum')
    # O comando 'scatter' soma os pesos separando-os pelo ID do stencil (batch)
    sum_per_stencil = scatter(pred_weights, batch, dim=0, reduce='sum')
    physics_loss = torch.mean((sum_per_stencil - target_sum)**2)
    
    # Loss Total
    return mse_loss 
    #+ lambda_reg * physics_loss






# ==========================================
# 3. LOOP DE TREINAMENTO (Exemplo Prático)
# ==========================================
if __name__ == "__main__":

    malhas = ['mesh2d-1'] 
    path = "../data_stencils/"
    dataset = load_data(malhas, path)
    
    model = StencilGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # ==========================================
    # 1. PREPARAÇÃO DOS DADOS (Antes do Treino)
    # ==========================================
    # Supondo que 'dataset' é a lista gerada pela função do passo anterior
    # (já com o filtro que removeu stencils de 1 elemento)

    # Vamos separar: 80% para treinar, 20% para testar
    tamanho_treino = int(0.8 * len(dataset))
    tamanho_teste = len(dataset) - tamanho_treino

    train_dataset, test_dataset = random_split(dataset, [tamanho_treino, tamanho_teste])

    # DataLoaders separados
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 facilita a análise individual

    print(f"Total de Stencils: {len(dataset)}")
    print(f"Treinamento: {len(train_dataset)} | Teste: {len(test_dataset)}")

    # --- AQUI VOCÊ RODA O SEU LOOP DE TREINAMENTO USANDO O train_loader ---
    model = StencilGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Iniciando Treinamento...")
    for epoch in range(8000):
        total_loss = 0
        for batch_data in train_loader:
            optimizer.zero_grad()
            
            # Forward Pass: A rede calcula os pesos para todos os nós no batch
            pred_weights = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            
            # Calcula o Loss (Exemplo para derivadas onde a soma é 0)
            loss = physics_informed_loss(pred_weights, batch_data.y, batch_data.batch, target_sum=1.0)
            
            # Backward Pass (Atualização de Pesos)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.5f}")
            
    print("Treinamento concluído. Modelo pronto para inferência ou exportação via TorchScript.")


    # ==========================================
    # 2. FASE DE TESTE / INFERÊNCIA
    # ==========================================
    print("\n--- INICIANDO TESTE NO CONJUNTO DE VALIDAÇÃO ---")

    # Coloca o modelo em modo de avaliação (desliga Dropout/BatchNorm se houver)
    model.eval() 

    erro_total_mse = 0
    erro_maximo_absoluto = 0.0
    violacao_fisica_media = 0.0
    target_sum = 0.0 # Mude para 1.0 se for interpolação de valor, 0.0 para derivada

    pesos_reais_lista = []
    pesos_preditos_lista = []

    # torch.no_grad() desliga o cálculo de gradientes (economiza memória e fica mais rápido)
    with torch.no_grad():
        for batch in test_loader:
            # 1. Faz a predição
            pred_weights = model(batch.x, batch.edge_index, batch.batch)
            true_weights = batch.y
            
            # 2. Calcula o Erro Médio Quadrático (MSE) do Stencil
            mse = F.mse_loss(pred_weights, true_weights)
            erro_total_mse += mse.item()
            
            # 3. Calcula o Erro Máximo (O mais importante para estabilidade do CFD!)
            erro_abs = torch.abs(pred_weights - true_weights)
            max_err_no_stencil = torch.max(erro_abs).item()
            if max_err_no_stencil > erro_maximo_absoluto:
                erro_maximo_absoluto = max_err_no_stencil
                
            # 4. Verifica a Física (Conservação)
            soma_predita = torch.sum(pred_weights).item()
            violacao_fisica_media += abs(soma_predita - target_sum)
            
            # Guarda os dados para plotar um gráfico no final
            pesos_reais_lista.extend(true_weights.tolist())
            pesos_preditos_lista.extend(pred_weights.tolist())

    # ==========================================
    # 3. RESULTADOS FINAIS
    # ==========================================
    numero_de_testes = len(test_loader)

    print(f"MSE Médio nos Testes:       {erro_total_mse / numero_de_testes:.6f}")
    print(f"Erro Máximo Absoluto:       {erro_maximo_absoluto:.6f}  <-- IMPORTANTE!")
    print(f"Erro Médio da Soma (Física):{violacao_fisica_media / numero_de_testes:.6f}")

    # ==========================================
    # 4. GRÁFICO VISUAL (Ground Truth vs Predição)
    # ==========================================
    plt.figure(figsize=(8, 8))
    plt.scatter(pesos_reais_lista, pesos_preditos_lista, alpha=0.5, color='blue')

    # Linha ideal (onde a predição é exatamente igual ao WLS analítico)
    min_val = min(min(pesos_reais_lista), min(pesos_preditos_lista))
    max_val = max(max(pesos_reais_lista), max(pesos_preditos_lista))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (Erro Zero)")

    plt.title("Teste da GNN: Pesos do WLS (Real vs Predito)")
    plt.xlabel("Pesos Exatos do WLS (HigFlow)")
    plt.ylabel("Pesos Previstos pela Rede Neural")
    plt.legend()
    plt.grid(True)
    plt.show()