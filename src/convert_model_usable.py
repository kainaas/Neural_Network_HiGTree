import torch
from MLP_structure import StencilMLP_Global

MAX_PONTOS = 60

caminho_modelo_in = "../models/mlp_relativo_30k.pt"

caminho_out = "../models/usable/mlp_relativo.pt"

checkpoint = torch.load(caminho_modelo_in)
model = StencilMLP_Global(max_n=MAX_PONTOS)

model.load_state_dict(checkpoint['modelo_state_dict'])

print("Iniciando exportação do modelo para C++...")

# 1. Mova o modelo para a CPU! 
# Isso evita aquele erro de "MPS" ou problemas se o C++ rodar em uma máquina sem GPU.
model.to('cpu')
model.double()
# 2. Coloque o modelo em modo de avaliação.
# Isso garante que camadas como Dropout e BatchNorm se comportem corretamente na inferência.
model.eval()

# 3. Converta o modelo para TorchScript.
# O JIT (Just-In-Time) compila a sua classe Python para um formato C++ nativo.
modelo_scriptado = torch.jit.script(model)

# 4. Salve o arquivo (por convenção, usamos a extensão .pt para modelos scriptados)
modelo_scriptado.save(caminho_out)

print("Modelo salvo com sucesso! O arquivo " + caminho_out + " está pronto para o LibTorch.")