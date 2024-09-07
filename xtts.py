import tkinter as tk
from tkinter import ttk, filedialog
from TTS.api import TTS
import os
from pydub import AudioSegment
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import shutil
import re  # Importe o módulo re para usar expressões regulares
import torch

MAX_THREADS = 1

def fragmentar_texto(texto, tamanho_maximo=200):
    """
    Fragmenta o texto em frases, respeitando a pontuação e o tamanho máximo,
    e substitui pontos finais por vírgulas nos fragmentos, exceto no último.
    """
    fragmentos = []
    fragmento_atual = ""
    
    i = 0
    while i < len(texto):
        caractere = texto[i]
        fragmento_atual += caractere

        if caractere in ('.', '!', '?', ';'):
            if len(fragmento_atual.strip()) > 0:  # Verifica se o fragmento não está vazio
                fragmentos.append(fragmento_atual.strip())
                fragmento_atual = ""
        elif i + 1 == len(texto) and fragmento_atual.strip():  # Verificação corrigida para o último fragmento
            # Adiciona o último fragmento, caso exista e não seja vazio
            fragmentos.append(fragmento_atual.strip())
        
        i += 1
    
    # Substitui '.' por ',' em todos os fragmentos, exceto o último
    for j in range(len(fragmentos) - 1):
        fragmentos[j] = fragmentos[j].replace('.', ',')
    
    return fragmentos

def atualizar_barra_progresso(total_fragmentos, processados_fragmentos, tempo_estimado=None, unindo_audios=False):
    """Atualiza a barra de progresso e a estimativa de tempo."""
    if unindo_audios:
        progresso = 90 + (10 * (processados_fragmentos / total_fragmentos))
        estimated_time_label["text"] = "Unindo áudios..."
    else:
        progresso = (processados_fragmentos / total_fragmentos) * 90
        if tempo_estimado is not None:
            minutos_restantes = int(tempo_estimado // 60)
            segundos_restantes = int(tempo_estimado % 60)
            estimated_time_label["text"] = f"Tempo estimado: {minutos_restantes:02d}:{segundos_restantes:02d}"
        else:
            estimated_time_label["text"] = "Calculando tempo..."

    progressbar["value"] = progresso
    progress_label["text"] = f"{processados_fragmentos}/{total_fragmentos} Fragmentos Criados"
    root.update_idletasks()

def gerar_fragmento_audio(fragmento, nome_arquivo, voz_clonada, idioma):
    """Gera um único fragmento de áudio e retorna o tempo de processamento."""
    inicio = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    
    # Obter a saída completa do xtts_v2
    outputs = tts.tts_to_file(
        text=fragmento,
        file_path=nome_arquivo,
        speaker_wav=[voz_clonada],
        language=idioma,
        split_sentences=True
    )

    fim = time.time()
    return fim - inicio

def gerar_audio_thread():
    """Função para executar a geração de áudio em uma thread separada."""
    global caminho_salvamento, total_fragmentos
    
    botao_iniciar.config(state="disabled")  # Desativa o botão
    texto = text_input.get("1.0", "end-1c")
    voz_clonada = voz_clonada_entry.get()
    idioma = idioma_combobox.get()

    # Obter o caminho completo de salvamento (pasta + nome do arquivo)
    caminho_salvamento = nome_arquivo_entry.get()

    fragmentos = fragmentar_texto(texto)
    total_fragmentos = len(fragmentos)

    # Criar a pasta "temp" se ela não existir, no mesmo diretório do script
    pasta_temp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    if not os.path.exists(pasta_temp):
        os.makedirs(pasta_temp)

    progressbar["maximum"] = total_fragmentos
    root.after(0, atualizar_barra_progresso, total_fragmentos, 0) # Inicia a barra de progresso

    tempos_de_processamento = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        for i, fragmento in enumerate(fragmentos):
            # Salvar arquivos temporários na pasta "temp"
            fragmento_arquivo = os.path.join(pasta_temp, f"temp_{i+1}.wav")
            future = executor.submit(gerar_fragmento_audio, fragmento, fragmento_arquivo, voz_clonada, idioma)
            futures.append(future)

        for i, future in enumerate(futures):
            try:
                tempo_fragmento = future.result()
                tempos_de_processamento.append(tempo_fragmento)
                tempo_medio = sum(tempos_de_processamento) / len(tempos_de_processamento) if tempos_de_processamento else 0
                tempo_estimado = tempo_medio * (total_fragmentos - (i + 1))
                root.after(0, atualizar_barra_progresso, total_fragmentos, i + 1, tempo_estimado)  # Atualiza na thread principal
            except Exception as e:
                print(f"Erro ao processar fragmento {i+1}: {e}")

    # Combina os arquivos de áudio na thread principal
    root.after(0, unir_audios)

def unir_audios():
    """Une os arquivos de áudio, removendo um tempo fixo do final de cada fragmento."""
    global caminho_salvamento, total_fragmentos
    pasta_temp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

    def atualizar_barra_juncao(i):
        """Função wrapper para atualizar a barra durante a junção."""
        atualizar_barra_progresso(total_fragmentos, i, unindo_audios=True)

    atualizar_barra_progresso(total_fragmentos, 0, unindo_audios=True)
    audio_combinado = None
    for i in range(total_fragmentos):
        fragmento_arquivo = os.path.join(pasta_temp, f"temp_{i+1}.wav")
        audio_fragmento = AudioSegment.from_wav(fragmento_arquivo)

        # Cortar 200ms do final do fragmento (ajuste este valor se necessário)
        audio_fragmento = audio_fragmento[:-250] 

        if audio_combinado is None:
            audio_combinado = audio_fragmento
        else:
            audio_combinado += audio_fragmento
        root.after(0, atualizar_barra_juncao, i+1)

    audio_combinado.export(f"{caminho_salvamento}", format="wav")
    shutil.rmtree(pasta_temp, ignore_errors=True)
    root.after(0, atualizar_barra_progresso, total_fragmentos, total_fragmentos, 0)
    botao_iniciar.config(state="normal")

def iniciar_geracao_audio():
    """Inicia a geração de áudio em uma nova thread."""
    threading.Thread(target=gerar_audio_thread).start()

def procurar_voz():
    arquivo = filedialog.askopenfilename(
        defaultextension=".wav",
        filetypes=[("WAV files", "*.wav")]
    )
    if arquivo:
        voz_clonada_entry.delete(0, tk.END)
        voz_clonada_entry.insert(0, arquivo)

def procurar_salvamento():
    arquivo = filedialog.asksaveasfilename(
        defaultextension=".wav",
        filetypes=[("WAV files", "*.wav")]
    )
    if arquivo:
        nome_arquivo_entry.delete(0, tk.END)
        nome_arquivo_entry.insert(0, arquivo)

# Interface gráfica
root = tk.Tk()
root.title("Leo Voicer")
root.geometry("600x460") 
root.configure(bg="#ffffff")

# Define o tamanho mínimo da janela (largura, altura)
root.minsize(600, 460) 

# Estilo para botões
style = ttk.Style()
style.configure("TButton", padding=6, relief="flat")

# Cabeçalho
header_frame = tk.Frame(root, bg="#ffffff")
header_frame.pack(fill="x")

title_label = tk.Label(header_frame, text="Leo Voicer", font=("Arial", 14, "bold"), bg="#ffffff")
title_label.pack(pady=10)

# Conteúdo principal
content_frame = tk.Frame(root, bg="#ffffff")
content_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Widgets
text_label = tk.Label(content_frame, text="Escreva o texto a ser dublado aqui:", bg="#ffffff")
text_label.grid(row=0, column=0, columnspan=4, sticky="w")

text_input = tk.Text(content_frame, height=5, width=50)
text_input.grid(row=1, column=0, columnspan=4, pady=(0, 10), sticky="ew")

progress_frame = tk.Frame(content_frame, bg="#ffffff")
progress_frame.grid(row=2, column=0, columnspan=4, sticky="ew") 

progress_label = tk.Label(progress_frame, text="0/0 Fragmentos Criados", bg="#ffffff")
progress_label.pack(side="left")

estimated_time_label = tk.Label(progress_frame, text="Tempo estimado: --:--", bg="#ffffff")
estimated_time_label.pack(side="right")

progressbar = ttk.Progressbar(content_frame, orient="horizontal", mode="determinate", length=400) 
progressbar.grid(row=3, column=0, columnspan=4, pady=(0, 10), sticky="ew") 

# Idioma
idioma_label = tk.Label(content_frame, text="Idioma:", bg="#ffffff")
idioma_label.grid(row=4, column=0, sticky="w")

idiomas = ['pt', 'en', 'es', 'fr', 'de', 'it', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko']
idioma_combobox = ttk.Combobox(content_frame, values=idiomas, state="readonly", width=5) 
idioma_combobox.set("pt")
idioma_combobox.grid(row=4, column=1, sticky="w") 

# Voz clonada
voz_clonada_label = tk.Label(content_frame, text="Localização da voz a ser clonada:", bg="#ffffff")
voz_clonada_label.grid(row=5, column=0, columnspan=2, sticky="w")

voz_clonada_entry = tk.Entry(content_frame)
voz_clonada_entry.grid(row=6, column=0, columnspan=3, sticky="ew")

botao_procurar_voz = ttk.Button(content_frame, text="Procurar", command=procurar_voz)
botao_procurar_voz.grid(row=6, column=3, sticky="ew")

# Salvamento
salvamento_label = tk.Label(content_frame, text="Localização e nome da voz dublada:", bg="#ffffff")
salvamento_label.grid(row=7, column=0, columnspan=2, sticky="w")

nome_arquivo_entry = tk.Entry(content_frame)
nome_arquivo_entry.grid(row=8, column=0, columnspan=3, sticky="ew")

botao_procurar_salvamento = ttk.Button(content_frame, text="Procurar", command=procurar_salvamento)
botao_procurar_salvamento.grid(row=8, column=3, sticky="ew")

# Botão Iniciar Transcrição
botao_iniciar = tk.Button(
    content_frame,
    text="Iniciar Transcrição",
    command=iniciar_geracao_audio,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12),
    padx=20,
    pady=10,
    relief="flat",
)
botao_iniciar.grid(row=9, column=0, columnspan=4, pady=(20, 0))

# Configurar o comportamento de redimensionamento das colunas
content_frame.columnconfigure(0, weight=1)
content_frame.columnconfigure(1, weight=0) 
content_frame.columnconfigure(2, weight=1) 
content_frame.columnconfigure(3, weight=0)

root.mainloop()
