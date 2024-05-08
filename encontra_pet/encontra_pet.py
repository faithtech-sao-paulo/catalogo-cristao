import gradio as gr
import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

# Função para carregar a imagem
def carregar_imagem(caminho_imagem):
    return cv2.imread(caminho_imagem)

# Função para calcular a assinatura visual da imagem
def calcular_assinatura(imagem):
    return cv2.resize(imagem, (32, 32)).flatten()

# Função para encontrar as 3 imagens mais similares
def encontrar_imagens_similares(imagem_entrada, pasta_imagens, n_similares=3):
    assinatura_entrada = calcular_assinatura(imagem_entrada)
    assinaturas_pasta = []
    nomes_imagens = []

    for arquivo in os.listdir(pasta_imagens):
        if arquivo.endswith(".jpg") or arquivo.endswith(".png"):
            caminho_imagem = os.path.join(pasta_imagens, arquivo)
            imagem = carregar_imagem(caminho_imagem)
            assinatura = calcular_assinatura(imagem)
            assinaturas_pasta.append(assinatura)
            nomes_imagens.append(arquivo)

    vizinhos = NearestNeighbors(n_neighbors=n_similares, algorithm='brute', metric='euclidean')
    vizinhos.fit(assinaturas_pasta)
    distancias, indices = vizinhos.kneighbors([assinatura_entrada])

    resultados = []
    for i in range(n_similares):
        nome_imagem = nomes_imagens[indices[0][i]]
        similaridade = 1 - distancias[0][i] / np.linalg.norm(assinatura_entrada)
        imagem_similar = carregar_imagem(os.path.join(pasta_imagens, nome_imagem))
        resultados.append((nome_imagem, similaridade, imagem_similar))

    return resultados

# Função para salvar a imagem com o nome fornecido pelo usuário e a data
def salvar_imagem(imagem, nome_imagem, pasta_destino):
    data_atual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nome_arquivo = f"{nome_imagem}_{data_atual}.jpg"
    caminho_destino = os.path.join(pasta_destino, nome_arquivo)
    cv2.imwrite(caminho_destino, imagem)

# Função para corrigir a imagem azulada
def corrigir_imagem_azulada(imagem):
    hsv_imagem = cv2.cvtColor(np.array(imagem, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hsv_imagem[:, :, 2] = cv2.multiply(hsv_imagem[:, :, 2], 0.8)
    corrigida = cv2.cvtColor(hsv_imagem, cv2.COLOR_HSV2RGB)
    return corrigida

# Interface Gradio
def interface_grafica(imagem, nome_imagem):
    pasta_destino = "C:/Users/jvict/Downloads/pets"
    
    # Extrair a imagem do componente Gradio Image
    imagem_array = np.array(imagem)
    
    imagem_corrigida = corrigir_imagem_azulada(imagem_array)
    salvar_imagem(imagem_corrigida, nome_imagem, pasta_destino)
    resultados = encontrar_imagens_similares(imagem_corrigida, pasta_destino)

    outputs = []
    for nome, similaridade, imagem_similar in resultados:
        outputs.append(gr.Image(imagem_similar, label=f"{nome} (Similaridade: {similaridade:.2f})"))

    return outputs

# Criar a interface
with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.Image(type="pil", label="Faça o Upload da Sua Imagem")
        text_input = gr.Textbox(label="Digite o Nome da Imagem")
    output = gr.Gallery()
    submit_button = gr.Button("Submit")
    submit_button.click(fn=interface_grafica, inputs=[image_input, text_input], outputs=output)

demo.launch()