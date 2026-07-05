import os
import pandas as pd
import ast
import torch
import numpy as np
from torch.utils.data import TensorDataset

def load_data_mlp(lista_nomes_malhas, path, MAX_N=60):
    """
    Carrega os dados de múltiplas malhas, rotaciona as faces, ordena os vizinhos 
    por distância e preenche com zeros (padding) para usar em um MLP.
    """
    todas_features = []
    todos_pesos = []
    todas_mascaras = []
    
    for nome_malha in lista_nomes_malhas:
        print(f"\nProcessando malha: {nome_malha}...")
        
        # 1. Carrega as coordenadas APENAS desta malha
        arquivo_coords = os.path.join(path, f"file_coords_{nome_malha}.csv")
        try:
            df_coords = pd.read_csv(arquivo_coords, index_col='gid')
        except FileNotFoundError:
            print(f"  [Aviso] Arquivo {arquivo_coords} não encontrado. Pulando malha.")
            continue
            

        arquivo_pesos = os.path.join(path, f"file_weights_{nome_malha}.txt")
        
        try:
            with open(arquivo_pesos, 'r') as f:
                dict_pesos = ast.literal_eval(f.read())
        except FileNotFoundError:
            print(f"  [Aviso] Arquivo {arquivo_pesos} não encontrado.")
            continue
            
        stencils_neste_arquivo = 0
            
        # 3. Constrói as entradas do MLP usando o df_coords local da malha atual
        for interpolado_coords, stencil_lista in dict_pesos.items():
            num_nos_no_stencil = len(stencil_lista)
            
                
            # Acesso seguro via loc usando o ID da malha atual
            cx = interpolado_coords[0]
            cy = interpolado_coords[1]
            
            pontos_temp = []
            
            for vizinho_id, peso_wls in stencil_lista:
                nx = df_coords.loc[vizinho_id, 'x']
                ny = df_coords.loc[vizinho_id, 'y']
                
                dx_rel = nx - cx
                dy_rel = ny - cy

                
                # Calcula a distância do centro ao vizinho para usar na ordenação
                distancia = np.sqrt(dx_rel**2 + dy_rel**2)
                
                # Guarda os dados temporariamente para podermos ordenar depois
                pontos_temp.append({
                    'dist': distancia,
                    'features': [dx_rel, dy_rel, 1.0/distancia], #Posição do ponto e o seu W da matriz do wls
                    'peso': peso_wls
                })
            
            # --- ORDENAÇÃO POR DISTÂNCIA ---
            # Garante que o vizinho mais próximo seja sempre o primeiro do vetor
            pontos_temp.sort(key=lambda p: p['dist'])

            # --- ZERO-PADDING (Preenchimento) ---
            features_stencil = []
            pesos_stencil = []
            mascara_stencil = []
            
            for i in range(MAX_N):
                if i < len(pontos_temp):
                    # Ponto real existente no estêncil
                    features_stencil.extend(pontos_temp[i]['features'])
                    pesos_stencil.append(pontos_temp[i]['peso'])
                    mascara_stencil.append(1.0) # 1.0 indica que o dado é real
                else:
                    # Falta ponto para atingir MAX_N -> Preenche com ZEROS
                    features_stencil.extend([0.0, 0.0, 0.0])
                    pesos_stencil.append(1.0)
                    mascara_stencil.append(0.0) # 0.0 diz para a função de Loss ignorar
            
            todas_features.append(features_stencil)
            todos_pesos.append(pesos_stencil)
            todas_mascaras.append(mascara_stencil)
            
            stencils_neste_arquivo += 1
                
        print(f"  -> Malha '{nome_malha}': {stencils_neste_arquivo} stencils úteis adicionados.")

    print(f"\nConcluído! Dataset final contém {len(todas_features)} stencils no total.")
    
    # 4. Converte as listas gigantes em Tensores do PyTorch
    X_tensor = torch.tensor(todas_features, dtype=torch.float32)
    Y_tensor = torch.tensor(todos_pesos, dtype=torch.float32)
    Mask_tensor = torch.tensor(todas_mascaras, dtype=torch.float32)
    
    # Empacota no formato nativo do PyTorch para facilitar o DataLoader
    return TensorDataset(X_tensor, Y_tensor, Mask_tensor)



def load_data_for_plot(lista_nomes_malhas, path):
    features_list = []
    weights_list = []
    
    for nome_malha in lista_nomes_malhas:
        print(f"\nProcessando malha: {nome_malha}...")
        
        # 1. Carrega as coordenadas APENAS desta malha
        arquivo_coords = os.path.join(path, f"file_coords_{nome_malha}.csv")
        try:
            df_coords = pd.read_csv(arquivo_coords, index_col='gid')
        except FileNotFoundError:
            print(f"  [Aviso] Arquivo {arquivo_coords} não encontrado. Pulando malha.")
            continue
            

        arquivo_pesos = os.path.join(path, f"file_weights_{nome_malha}.txt")
        
        try:
            with open(arquivo_pesos, 'r') as f:
                dict_pesos = ast.literal_eval(f.read())
        except FileNotFoundError:
            print(f"  [Aviso] Arquivo {arquivo_pesos} não encontrado.")
            continue
            
        stencils_neste_arquivo = 0
            
        # 3. Constrói as entradas do MLP usando o df_coords local da malha atual
        for interpolado_coords, stencil_lista in dict_pesos.items():
            num_nos_no_stencil = len(stencil_lista)
            
                
            # Acesso seguro via loc usando o ID da malha atual
            cx = interpolado_coords[0]
            cy = interpolado_coords[1]
            
            pontos_temp = []
            
            for vizinho_id, peso_wls in stencil_lista:
                nx = df_coords.loc[vizinho_id, 'x']
                ny = df_coords.loc[vizinho_id, 'y']
                
                dx_rel = nx - cx
                dy_rel = ny - cy

                
                # Calcula a distância do centro ao vizinho para usar na ordenação
                distancia = np.sqrt(dx_rel**2 + dy_rel**2)
                
                # Guarda os dados temporariamente para podermos ordenar depois
                pontos_temp.append({
                    'features': [dx_rel, dy_rel, distancia, 1.0/distancia], #Posição do ponto, distância e o seu W da matriz do wls
                    'peso': peso_wls
                })
            
            # --- ORDENAÇÃO POR DISTÂNCIA ---
            # Garante que o vizinho mais próximo seja sempre o primeiro do vetor
            pontos_temp.sort(key=lambda p: p['features'][2])
            
            for i in range(len(pontos_temp)):
                features_list.append(pontos_temp[i]['features'])
                weights_list.append(pontos_temp[i]['peso'])

            stencils_neste_arquivo += 1
                
        print(f"  -> Malha '{nome_malha}': {stencils_neste_arquivo} stencils úteis adicionados.")

    return features_list, weights_list


# ==========================================
# COMO USAR:
# ==========================================
if __name__ == "__main__":
    # Basta listar o sufixo das malhas que você extraiu do HigFlow
    minhas_malhas = ['mesh2d-1'] 
    path = "../data_stencils/"
    # O script vai processar 4 arquivos txt e 1 csv para CADA malha da lista
    dataset = load_data_mlp(minhas_malhas, path)
    print(dataset)