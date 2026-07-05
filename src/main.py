import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_treat import load_data_mlp
import torch.nn.functional as F
import matplotlib.pyplot as plt
import MLP_structure as MLP

path_models = "models/"
path_data = "data/"

grids = ['mesh2d-1', 'mesh00000001-2d', 'mesh00000002-2d', 'bfs-mesh-2-2d', 'mesh00000003-2d'] 
path_grids = "../data_stencils/"


treinar = False

carregar_modelo = True
model_in = path_models + "mlp_relativo.pt"

carregar_dados = True
train_in = path_data + "data_treino_rel.pth"
test_in = path_data + "data_teste_rel.pth"

salvar_modelo = False
model_out = path_models + "mlp_relativo.pt"

salvar_dados = False
train_out = path_data + "data_treino_rel.pth"
test_out = path_data + "data_teste_rel.pth"

epochs = 3000

batches_size = 128

MAX_PONTOS = 60 #maior estêncil que encontrar no HigFlow





#========================================================================
# 1) INICIALIZANDO DADOS E MODELO
#========================================================================
model = MLP.StencilMLP_Global(max_n=MAX_PONTOS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

if carregar_modelo == True:
    checkpoint = torch.load(model_in)

    model.load_state_dict(checkpoint['modelo_state_dict'])
    optimizer.load_state_dict(checkpoint['otimizador_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print("Model loaded")


dataset = 0

tamanho_treino = 0
tamanho_teste = 0

test_dataset = 0
train_dataset = 0

train_loader = 0
test_loader = 0


if carregar_dados == True:
    # Carrega os dicionários salvos
    dados_teste = torch.load(test_in)
    dados_treino = torch.load(train_in)

    # Remonta o TensorDataset
    test_dataset = TensorDataset(dados_teste['x'], dados_teste['y'], dados_teste['mask'])
    train_dataset = TensorDataset(dados_treino['x'], dados_treino['y'], dados_treino['mask'])

    # Refaz o DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batches_size, shuffle=False)

    print(f"Dados de teste carregados! Total de estênceis: {len(test_dataset) + len(train_dataset)}")
    
else:
    dataset = load_data_mlp(grids, path_grids, MAX_N=MAX_PONTOS)
    # Vamos separar: 80% para treinar, 20% para testar
    tamanho_treino = int(0.8 * len(dataset))
    tamanho_teste = len(dataset) - tamanho_treino

    train_dataset, test_dataset = random_split(dataset, [tamanho_treino, tamanho_teste])

    # DataLoaders separados
    train_loader = DataLoader(train_dataset, batch_size=batches_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 facilita a análise individual
    print(f"Total de Stencils: {len(dataset)}")

print(f"Treinamento: {len(train_dataset)} | Teste: {len(test_dataset)}")


if salvar_dados == True:
    # ==========================================
    # SALVANDO OS DATASETS DE TREINO E TESTE
    # ==========================================
    print("\nSalvando os datasets divididos no disco...")

    # O 'dataset' original é um TensorDataset, então ele possui o atributo .tensors
    X_completo, Y_completo, Mask_completo = 0,0,0
    if carregar_dados:
        torch.save(train_dataset, train_out)
        torch.save(test_dataset, test_out)
    else:
        X_completo, Y_completo, Mask_completo = dataset.tensors

        # Extrai os índices sorteados pelo random_split
        idx_treino = train_dataset.indices
        idx_teste = test_dataset.indices

        # Salva o pacote de treino
        torch.save({
            'x': X_completo[idx_treino],
            'y': Y_completo[idx_treino],
            'mask': Mask_completo[idx_treino]
        }, train_out)

        # Salva o pacote de teste
        torch.save({
            'x': X_completo[idx_teste],
            'y': Y_completo[idx_teste],
            'mask': Mask_completo[idx_teste]
        }, test_out)

    print("Datasets salvos com sucesso: 'dataset_treino.pth' e 'dataset_teste.pth'")



best_test_model = model_out[0:model_out.rfind(".")] + "_best" + model_out[model_out.rfind("."):]
best_test = -1.0
#========================================================================
# 2) FASE DE TREINO
#========================================================================
if treinar == True:
    print("Iniciando Treino do MLP com Ordenação por Distância...")
    for epoch in range(epochs+1):
        model.train()
        total_loss = 0
        
        # O loader agora retorna 3 variáveis: Entrada(X), Gabarito(Y) e Máscara
        for batch_x, batch_y, batch_mask in train_loader:
            optimizer.zero_grad()
            
            # Faz a predição de todo o estêncil de uma vez
            pred_weights = model(batch_x)
            
            loss = 0.0
            # Calcula a loss aplicando a máscara para ignorar os zeros inseridos
            loss = MLP.masked_loss_relative(pred_weights, batch_y, batch_mask)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        loss_media = total_loss / len(train_loader)
        scheduler.step(loss_media)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss_media:.6f}")
            
            #Coloca o modelo em modo de avaliação
            model.eval() 
            
            # 2. Inicializa a soma do teste FORA do loop
            test_loss_total = 0.0 
            
            with torch.no_grad():
                for test_x, test_y, test_mask in test_loader:
                    
                    # 3. FAZ A PREVISÃO REAL NOS DADOS DE TESTE
                    test_pred = model(test_x) 
                    
                    loss_teste = MLP.masked_loss_relative(test_pred, test_y, test_mask)
                        
                    test_loss_total += loss_teste.item()
            
            # Calcula a média real do conjunto de teste inteiro
            test_loss_media = test_loss_total / len(test_loader)
            print(f"          | Test Loss:  {test_loss_media:.6f} \n")
            
            # Avalia se é o melhor modelo
            if best_test < 0 or test_loss_media <= best_test:
                best_test = test_loss_media
                if salvar_modelo:
                    checkpoint = {
                        'modelo_state_dict': model.state_dict(),
                        'otimizador_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }
                    torch.save(checkpoint, best_test_model)
                    print(f"          -> Novo melhor modelo salvo!")



# ==========================================
# SALVANDO O MODELO PARA PYTHON
# ==========================================
if salvar_modelo == True:
    checkpoint = {
        'modelo_state_dict': model.state_dict(),
        'otimizador_state_dict': optimizer.state_dict(), # <- Salva a memória do Adam!
        'scheduler_state_dict': scheduler.state_dict()   # <- Salva o ritmo do Scheduler!
    }
    torch.save(checkpoint, model_out)


# ==========================================
# 3) FASE DE TESTE / INFERÊNCIA
# ==========================================
print("\n--- INICIANDO TESTE NO CONJUNTO DE VALIDAÇÃO ---")

# Coloca o modelo em modo de avaliação (desliga Dropout/BatchNorm se houver)
model.eval() 

erro_total_mse = 0
erro_maximo_absoluto = 0.0
erro_relativo_quadratico = 0.0
erro_maximo_relativo = 0.0
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

        #5. Calcula o erro quadrático médio relativo
        erro_relativo_quadratico = MLP.masked_loss_relative(pred_weights, batch_y, batch_mask).item()

        #6. Calcula o máximo do erro quadrático relativo
        erro_relativo = torch.abs((pred_validos - true_validos)**2) / true_validos
        max_relativo_no_stencil = torch.max(erro_relativo).item()
        if max_relativo_no_stencil > erro_maximo_relativo:
            erro_maximo_relativo = max_relativo_no_stencil

        # 6. Verifica a Física (Conservação)
        # Como o batch_size do test_loader é 1, podemos somar direto os válidos
        soma_predita = torch.sum(pred_validos).item()
        violacao_fisica_media += abs(soma_predita - target_sum)
        
        # Guarda os dados para plotar o gráfico (apenas os válidos!)
        pesos_reais_lista.extend(true_validos.tolist())
        pesos_preditos_lista.extend(pred_validos.tolist())

# ==========================================
# 4) RESULTADOS FINAIS
# ==========================================
numero_de_testes = len(test_loader)

print(f"MSE Médio nos Testes:       {erro_total_mse / numero_de_testes:.6f}")
print(f"Erro Máximo Absoluto:       {erro_maximo_absoluto:.6f}")
print(f"MSE médio relativo:         {erro_relativo_quadratico:.6f}")
print(f"MSE Máximo relativo:        {erro_maximo_relativo:.6f}")
print(f"Erro Médio da Soma (Física):{violacao_fisica_media / numero_de_testes:.6f}")



# ==========================================
# 4. GRÁFICO VISUAL (Ground Truth vs Predição)
# ==========================================
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(221)
ax.scatter(pesos_reais_lista, pesos_preditos_lista, alpha=0.5, color='blue')

# Linha ideal (onde a predição é exatamente igual ao WLS analítico)
if len(pesos_reais_lista) > 0 and len(pesos_preditos_lista) > 0:
    min_val = min(min(pesos_reais_lista), min(pesos_preditos_lista))
    max_val = max(max(pesos_reais_lista), max(pesos_preditos_lista))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (Erro Zero)")

ax.set_title("Teste do MLP: Pesos do WLS (Real vs Predito)")
ax.set_xlabel("Pesos Exatos do WLS (HigFlow)")
ax.set_ylabel("Pesos Previstos pelo MLP")
ax.legend()
ax.grid(True)

ax2 = fig.add_subplot(222)
ax2.set_title("Erro relativo em função do valor real do peso")
ax2.set_xlabel("Peso real")
ax2.set_ylabel("Erro")
ax2.scatter(pesos_reais_lista, np.abs(np.array(pesos_preditos_lista) - np.array(pesos_reais_lista)/np.array(pesos_preditos_lista)))

ax2 = fig.add_subplot(223)
ax2.set_title("Erro em função do valor real do peso")
ax2.set_xlabel("Peso real")
ax2.set_ylabel("Erro")
ax2.scatter(pesos_reais_lista, np.abs(np.array(pesos_preditos_lista) - np.array(pesos_reais_lista)))

plt.show()