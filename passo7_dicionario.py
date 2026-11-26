import gzip
import json
import os
import random
import time
import urllib.request
import zipfile
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
random.seed(42)

# =============================================================
# PASSO 7 - Ensinar o mini-GPT com um DICIONARIO (Wikcionario)
# -------------------------------------------------------------
# Fonte: Wikcionario em portugues (licenca livre CC BY-SA),
# extraido pelo projeto Kaikki (kaikki.org).
# =============================================================

# 1) BAIXA O DICIONARIO (so na primeira vez, ~35 MB)
URL = "https://kaikki.org/dictionary/downloads/pt/pt-extract.jsonl.gz"
ARQUIVO_GZ = "pt-extract.jsonl.gz"
ARQUIVO_TSV = "dicionario.tsv"   # versao "limpa" e pequena, feita por nos

if not os.path.exists(ARQUIVO_TSV):
    if not os.path.exists(ARQUIVO_GZ):
        print("Baixando o Wikcionario (35 MB, pode levar 1 minuto)...")
        try:
            pedido = urllib.request.Request(
                URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(pedido) as r, open(ARQUIVO_GZ, "wb") as f:
                f.write(r.read())
        except Exception as erro:
            print("Nao consegui baixar:", erro)
            print("Baixe pelo navegador e coloque nesta pasta:", URL)
            raise SystemExit

    # Le o arquivo gigante e guarda so o que interessa:
    # palavras em portugues + a primeira definicao, curta
    print("Lendo o dicionario e separando as definicoes...")
    verbetes = {}
    with gzip.open(ARQUIVO_GZ, "rt", encoding="utf-8") as f:
        for linha in f:
            item = json.loads(linha)
            if item.get("lang_code") != "pt":
                continue
            palavra = item.get("word", "")
            if not palavra.isalpha() or len(palavra) > 12 or palavra in verbetes:
                continue
            for sentido in item.get("senses", []):
                glosses = sentido.get("glosses") or []
                if not glosses:
                    continue
                definicao = glosses[0].strip().rstrip(".;:")
                if 8 <= len(definicao) <= 45 and "\n" not in definicao:
                    verbetes[palavra] = definicao[0].upper() + definicao[1:] + "."
                    break
    with open(ARQUIVO_TSV, "w", encoding="utf-8") as f:
        for palavra, definicao in verbetes.items():
            f.write(f"{palavra}\t{definicao}\n")
    print(f"Pronto: {len(verbetes):,} verbetes salvos em {ARQUIVO_TSV}\n")

verbetes = []
with open(ARQUIVO_TSV, encoding="utf-8") as f:
    for linha in f:
        palavra, definicao = linha.rstrip("\n").split("\t")
        verbetes.append((palavra, definicao))
random.shuffle(verbetes)

# Separa 100 palavras que a IA NUNCA vai ver, para testar depois
prova = verbetes[:100]
estudo = verbetes[100:]
print(f"{len(estudo):,} palavras para estudar e {len(prova)} guardadas para a prova\n")

# 2) CARREGA O MINI-GPT QUE JA SABE CONVERSAR (passo 6)
salvo = torch.load("mini_gpt.pt")
chars = salvo["chars"]
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for c, i in ctoi.items()}
V = len(chars)
CONTEXTO, EMB, CABECAS, CAMADAS = 64, 128, 4, 4

def codificar(texto):
    return [ctoi.get(c, ctoi[" "]) for c in texto]

class Atencao(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMB, 3 * EMB)
        self.saida = nn.Linear(EMB, EMB)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(EMB, dim=2)
        q = q.view(B, T, CABECAS, -1).transpose(1, 2)
        k = k.view(B, T, CABECAS, -1).transpose(1, 2)
        v = v.view(B, T, CABECAS, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.saida(y.transpose(1, 2).reshape(B, T, EMB))

class Bloco(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMB)
        self.atencao = Atencao()
        self.norm2 = nn.LayerNorm(EMB)
        self.pensar = nn.Sequential(
            nn.Linear(EMB, 4 * EMB), nn.GELU(), nn.Linear(4 * EMB, EMB))

    def forward(self, x):
        x = x + self.atencao(self.norm1(x))
        x = x + self.pensar(self.norm2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_char = nn.Embedding(V, EMB)
        self.emb_pos = nn.Embedding(CONTEXTO, EMB)
        self.blocos = nn.Sequential(*[Bloco() for _ in range(CAMADAS)])
        self.norm = nn.LayerNorm(EMB)
        self.cabeca = nn.Linear(EMB, V)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb_char(x) + self.emb_pos(torch.arange(T))
        return self.cabeca(self.norm(self.blocos(h)))

modelo = MiniGPT()

# 3) MONTA OS DADOS DE TREINO
def formatar(pergunta, resposta):
    return f"Você: {pergunta}\nIA: {resposta}\n\n"

def texto_dicionario(lista, repeticoes):
    partes = []
    for _ in range(repeticoes):
        random.shuffle(lista)
        partes += [formatar(f"O que é {p}?", d) for p, d in lista]
    return "".join(partes)

conversas_basicas = [
    ("Oi!", "Olá! Em que posso ajudar?"),
    ("Quem é você?", "Sou o mini-GPT, uma IA criada pelo Octavio."),
    ("Qual é o seu nome?", "Meu nome é mini-GPT."),
    ("Obrigado!", "De nada! Precisando, é só chamar."),
    ("Tchau!", "Tchau! Até a próxima."),
]
dados_dic = torch.tensor(codificar(texto_dicionario(estudo, 3)))
dados_conv = torch.tensor(codificar(
    "".join(formatar(p, r) for p, r in conversas_basicas * 200)))

nltk.download("machado", quiet=True)
zip_path = str(nltk.data.find("corpora/machado.zip"))
machado = ""
with zipfile.ZipFile(zip_path) as z:
    for nome in sorted(z.namelist()):
        if "/romance/" in nome and nome.endswith(".txt"):
            machado += z.read(nome).decode("cp1252", errors="replace")
dados_machado = torch.tensor(codificar(machado.replace("\r", "")))

def pegar_lote(fonte, tamanho=32):
    pos = torch.randint(0, len(fonte) - CONTEXTO - 1, (tamanho,))
    x = torch.stack([fonte[p:p + CONTEXTO] for p in pos])
    y = torch.stack([fonte[p + 1:p + CONTEXTO + 1] for p in pos])
    return x, y

# 4) TREINO (so na primeira vez)
if os.path.exists("mini_gpt_dicionario.pt"):
    modelo.load_state_dict(torch.load("mini_gpt_dicionario.pt"))
    print("Modelo com dicionario carregado (mini_gpt_dicionario.pt)\n")
else:
    if os.path.exists("mini_gpt_chat.pt"):
        modelo.load_state_dict(torch.load("mini_gpt_chat.pt"))
    else:
        modelo.load_state_dict(salvo["modelo"])
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=5e-4)
    PASSOS = 3000
    inicio = time.time()
    print("Estudando o dicionario...")
    for passo in range(PASSOS + 1):
        sorteio = random.random()
        if sorteio < 0.7:
            fonte = dados_dic        # 70% dicionario
        elif sorteio < 0.85:
            fonte = dados_conv       # 15% conversas basicas
        else:
            fonte = dados_machado    # 15% Machado
        x, y = pegar_lote(fonte)
        perda = F.cross_entropy(modelo(x).view(-1, V), y.view(-1))
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
        if passo % 250 == 0:
            with torch.no_grad():
                xd, yd = pegar_lote(dados_dic, 128)
                erro = F.cross_entropy(modelo(xd).view(-1, V), yd.view(-1))
            print(f"passo {passo:4d} | erro no dicionario: {erro.item():.3f}"
                  f" | {time.time() - inicio:.0f}s")
    torch.save(modelo.state_dict(), "mini_gpt_dicionario.pt")
    print("\nModelo salvo em mini_gpt_dicionario.pt\n")

modelo.eval()

@torch.no_grad()
def responder(pergunta, temperatura=0.3, mostrar=True):
    ids = torch.tensor([codificar(f"Você: {pergunta}\nIA:")])
    resposta = ""
    for _ in range(150):
        logits = modelo(ids[:, -CONTEXTO:])[:, -1, :] / temperatura
        prox = torch.multinomial(F.softmax(logits, dim=-1), 1)
        letra = itoc[prox.item()]
        if letra == "\n":
            break
        resposta += letra
        if mostrar:
            print(letra, end="", flush=True)
        ids = torch.cat([ids, prox], dim=1)
    if mostrar:
        print()
    return resposta.strip()

# 5) A PROVA: palavras estudadas x palavras nunca vistas
def prova_rapida(titulo, lista):
    print(titulo)
    for palavra, definicao in lista[:5]:
        print(f"  {palavra}")
        print(f"     Dicionario: {definicao}")
        print(f"     mini-GPT:   {responder(f'O que é {palavra}?', mostrar=False)}")
    print()

prova_rapida("PALAVRAS QUE ELE ESTUDOU:", estudo)
prova_rapida("PALAVRAS QUE ELE NUNCA VIU:", prova)

# 6) CONVERSAR
temperatura = 0.3
print("=" * 60)
print(" Pergunte o significado de uma palavra. Ex: O que é saudade?")
print(" Comandos:  /temp 0.8   /prova (mostra palavras que ele nunca viu)   /sair")
print("=" * 60)
while True:
    try:
        pergunta = input("\nVoce: ").strip()
    except EOFError:
        break
    if pergunta == "/sair":
        break
    if pergunta == "/prova":
        print("Palavras que ele nunca viu:", ", ".join(p for p, _ in prova[:20]))
        continue
    if pergunta.startswith("/temp"):
        try:
            temperatura = float(pergunta.split()[1])
            print(f"Temperatura agora e {temperatura}")
        except (IndexError, ValueError):
            print("Use assim: /temp 0.8")
        continue
    if pergunta:
        print("IA:", end="", flush=True)
        responder(pergunta, temperatura)
