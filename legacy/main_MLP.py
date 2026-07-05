import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_treat import load_data_mlp
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

import torch.nn as nn

class StencilMLP_Global(nn.Module):
    def __init__(self, max_n=20, features_por_no=4, hidden_dim=256): # Aumentei para 256
        super(StencilMLP_Global, self).__init__()
        
        tamanho_entrada = max_n * features_por_no
        
        self.net = nn.Sequential(
            # Camada 1: Entrada -> Oculta 1
            nn.Linear(tamanho_entrada, hidden_dim),
            nn.LeakyReLU(0.1),
            
            # Camada 2: Oculta 1 -> Oculta 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            
            # --- NOVAS CAMADAS ADICIONADAS AQUI ---
            # Camada 3: Mantém a dimensão (aprofunda o raciocínio)
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            
            # Camada 4: Começa a reduzir (funil de extração de features)
            nn.Linear(hidden_dim, hidden_dim // 2), # 256 -> 128
            nn.LeakyReLU(0.1),
            # --------------------------------------
            
            # Camada 5: Reduz mais um pouco
            nn.Linear(hidden_dim // 2, hidden_dim // 4), # 128 -> 64
            nn.LeakyReLU(0.1),
            
            # Camada 6 (Saída): Oculta 5 -> Previsão final
            nn.Linear(hidden_dim // 4, max_n)            # 64 -> MAX_N
        )

    def forward(self, x):
        # x já entra achatado (flattened) no formato [Batch, MAX_N * 4]
        pesos_preditos = self.net(x)
        return pesos_preditos
    

def masked_physics_loss(pred_weights, true_weights, mask, target_sum=1.0):
    # FATOR DE AMPLIFICAÇÃO (Lupa)
    # Multiplicamos por 100 para o erro de 0.06 virar 6.0 e a rede "sentir a dor"
    escala = 100.0 
    
    pred_amplificado = pred_weights * escala
    true_amplificado = true_weights * escala
    
    # 1. Erro Absoluto (L1) com a lupa
    erro_absoluto = torch.abs((pred_amplificado - true_amplificado)**2)
    erro_mascarado = erro_absoluto * mask
    
    # Média apenas dos pontos válidos
    mae_loss = torch.sum(erro_mascarado) / torch.sum(mask)
    
    # 2. Conservação Física (DESLIGADA TEMPORARIAMENTE)
    # pesando por 0.0 para garantir que ela não trapaceie chutando zeros
    pesos_validos = pred_weights * mask
    soma = torch.sum(pesos_validos, dim=1)
    physics_loss = torch.mean(torch.abs(soma - target_sum))
    
    return mae_loss + (0.05 * physics_loss)




caminho_modelo_in = "modelo_mlp_pesos.pth"

caminho_modelo_out = "modelo_mlp_pesos.pth"

treinar = True

carregar_modelo = False

carregar_dados = False

salvar_modelo = True

salvar_dados = True

epochs = 20000

MAX_PONTOS = 70 # Ajuste de acordo com o maior estêncil que encontrar no HigFlow

model = StencilMLP_Global(max_n=MAX_PONTOS)
if carregar_modelo == True:
    model.load_state_dict(torch.load(caminho_modelo_in))


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)


malhas = ['mesh2d-1', 'mesh00000001-2d', 'mesh00000002-2d', 'bfs-mesh-1-2d', 'bfs-mesh-2-2d', 'mesh00000003-2d'] 
path = "../data_stencils/"
dataset = load_data_mlp(malhas, path, MAX_N=MAX_PONTOS)



tamanho_treino = 0
tamanho_teste = 0

test_dataset = 0
train_dataset = 0

train_loader = 0
test_loader = 0

if treinar == True:
    # ==========================================
    # 1. PREPARAÇÃO DOS DADOS (Antes do Treino)
    # ==========================================
    # Supondo que 'dataset' é a lista gerada pela função do passo anterior
    # (já com o filtro que removeu stencils de 1 elemento)

    if carregar_dados == True:
        # Carrega os dicionários salvos
        dados_teste = torch.load("dataset_teste.pth")
        dados_treino = torch.load("dataset_treino.pth")

        # Remonta o TensorDataset
        test_dataset = TensorDataset(dados_teste['x'], dados_teste['y'], dados_teste['mask'])
        train_dataset = TensorDataset(dados_treino['x'], dados_treino['y'], dados_treino['mask'])

        # Refaz o DataLoader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        print(f"Dados de teste carregados! Total de estênceis: {len(test_dataset)}")
        
    else:
        # Vamos separar: 80% para treinar, 20% para testar
        tamanho_treino = int(0.8 * len(dataset))
        tamanho_teste = len(dataset) - tamanho_treino

        train_dataset, test_dataset = random_split(dataset, [tamanho_treino, tamanho_teste])

        # DataLoaders separados
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 facilita a análise individual

    print(f"Total de Stencils: {len(dataset)}")
    print(f"Treinamento: {len(train_dataset)} | Teste: {len(test_dataset)}")


    if salvar_dados == True:
        # ==========================================
        # SALVANDO OS DATASETS DE TREINO E TESTE
        # ==========================================
        print("\nSalvando os datasets divididos no disco...")

        # O 'dataset' original é um TensorDataset, então ele possui o atributo .tensors
        X_completo, Y_completo, Mask_completo = dataset.tensors

        # Extrai os índices sorteados pelo random_split
        idx_treino = train_dataset.indices
        idx_teste = test_dataset.indices

        # Salva o pacote de treino
        torch.save({
            'x': X_completo[idx_treino],
            'y': Y_completo[idx_treino],
            'mask': Mask_completo[idx_treino]
        }, "dataset_treino.pth")

        # Salva o pacote de teste
        torch.save({
            'x': X_completo[idx_teste],
            'y': Y_completo[idx_teste],
            'mask': Mask_completo[idx_teste]
        }, "dataset_teste.pth")

        print("Datasets salvos com sucesso: 'dataset_treino.pth' e 'dataset_teste.pth'")

    print("Iniciando Treino do MLP com Ordenação por Distância...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # O loader agora retorna 3 variáveis: Entrada(X), Gabarito(Y) e Máscara
        for batch_x, batch_y, batch_mask in train_loader:
            optimizer.zero_grad()
            
            # Faz a predição de todo o estêncil de uma vez
            pred_weights = model(batch_x)
            
            # Calcula a loss aplicando a máscara para ignorar os zeros inseridos
            loss = masked_physics_loss(pred_weights, batch_y, batch_mask, target_sum=1.0)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        loss_media = total_loss / len(train_loader)
        scheduler.step(loss_media)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss_media:.6f}")

else:
    if carregar_dados:
        # Carrega os dicionários salvos
        dados_teste = torch.load("dataset_teste.pth")

        # Remonta o TensorDataset
        test_dataset = TensorDataset(dados_teste['x'], dados_teste['y'], dados_teste['mask'])

        # Refaz o DataLoader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        print(f"Dados de teste carregados! Total de estênceis: {len(test_dataset)}")
    tamanho_teste = len(dataset)
    test_dataset = dataset
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 facilita a análise individual

# ==========================================
# 2. FASE DE TESTE / INFERÊNCIA
# ==========================================
print("\n--- INICIANDO TESTE NO CONJUNTO DE VALIDAÇÃO ---")

# Coloca o modelo em modo de avaliação (desliga Dropout/BatchNorm se houver)
model.eval() 

erro_total_mse = 0
erro_maximo_absoluto = 0.0
violacao_fisica_media = 0.0
target_sum = 1.0 # Mude para 1.0 se for interpolação de valor, 0.0 para derivada

pesos_reais_lista = []
pesos_preditos_lista = []

# torch.no_grad() desliga o cálculo de gradientes (economiza memória e fica mais rápido)
with torch.no_grad():
    # AGORA DESEMPACOTAMOS X, Y e MASK
    for batch_x, batch_y, batch_mask in test_loader:
        
        # 1. Faz a predição do estêncil inteiro (com os zeros junto)
        pred_weights = model(batch_x)
        
        # 2. FILTRA OS DADOS REAIS USANDO A MÁSCARA
        # Cria um filtro booleano onde a máscara é 1
        filtro_real = batch_mask > 0.5 
        
        # Extrai apenas os pesos reais (ignora o padding)
        pred_validos = pred_weights[filtro_real]
        true_validos = batch_y[filtro_real]
        
        # Se por algum motivo o estêncil for vazio (não deveria), pula
        if len(pred_validos) == 0:
            continue
            
        # 3. Calcula o Erro Médio Quadrático (MSE) APENAS dos pontos válidos
        mse = F.mse_loss(pred_validos, true_validos)
        erro_total_mse += mse.item()
        
        # 4. Calcula o Erro Máximo Absoluto
        erro_abs = torch.abs(pred_validos - true_validos)
        max_err_no_stencil = torch.max(erro_abs).item()
        if max_err_no_stencil > erro_maximo_absoluto:
            erro_maximo_absoluto = max_err_no_stencil
            
        # 5. Verifica a Física (Conservação)
        # Como o batch_size do test_loader é 1, podemos somar direto os válidos
        soma_predita = torch.sum(pred_validos).item()
        violacao_fisica_media += abs(soma_predita - target_sum)
        
        # Guarda os dados para plotar o gráfico (apenas os válidos!)
        pesos_reais_lista.extend(true_validos.tolist())
        pesos_preditos_lista.extend(pred_validos.tolist())

# ==========================================
# 3. RESULTADOS FINAIS
# ==========================================
numero_de_testes = len(test_loader)

print(f"MSE Médio nos Testes:       {erro_total_mse / numero_de_testes:.6f}")
print(f"Erro Máximo Absoluto:       {erro_maximo_absoluto:.6f}  <-- IMPORTANTE!")
print(f"Erro Médio da Soma (Física):{violacao_fisica_media / numero_de_testes:.6f}")




# ==========================================
# SALVANDO O MODELO PARA PYTHON
# ==========================================
if salvar_modelo == True:
    torch.save(model.state_dict(), caminho_modelo_out)
    print(f"Modelo salvo para Python em: {caminho_modelo_out}")



# ==========================================
# 4. GRÁFICO VISUAL (Ground Truth vs Predição)
# ==========================================
plt.figure(figsize=(8, 8))
plt.scatter(pesos_reais_lista, pesos_preditos_lista, alpha=0.5, color='blue')

# Linha ideal (onde a predição é exatamente igual ao WLS analítico)
if len(pesos_reais_lista) > 0 and len(pesos_preditos_lista) > 0:
    min_val = min(min(pesos_reais_lista), min(pesos_preditos_lista))
    max_val = max(max(pesos_reais_lista), max(pesos_preditos_lista))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (Erro Zero)")

plt.title("Teste do MLP: Pesos do WLS (Real vs Predito)")
plt.xlabel("Pesos Exatos do WLS (HigFlow)")
plt.ylabel("Pesos Previstos pelo MLP")
plt.legend()
plt.grid(True)
plt.show()