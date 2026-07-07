import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import MLP_structure as MLP

def inference(model: nn.Module, 
              test_loader: DataLoader
              ) -> tuple[
                   float,
                   float,
                   float,
                   float,
                   float,
                   list,
                   list]:
    model.eval() 

    erro_total_mse = 0
    erro_maximo_absoluto = 0.0
    erro_relativo_quadratico = 0.0
    erro_maximo_relativo = 0.0
    violacao_fisica_media = 0.0
    target_sum = 1.0

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
            erro_relativo = torch.abs((pred_validos - true_validos)**2 / true_validos)
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
    return erro_total_mse, erro_maximo_absoluto, erro_relativo_quadratico, erro_maximo_relativo, violacao_fisica_media, pesos_reais_lista, pesos_preditos_lista