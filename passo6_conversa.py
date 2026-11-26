import os
import random
import time
import zipfile
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
random.seed(42)

# =============================================================
# PASSO 6 - Ensinar o mini-GPT a CONVERSAR (fine-tuning)
# -------------------------------------------------------------
# Pega o modelo do passo 4 (que so sabe continuar texto) e treina
# mais um pouco com exemplos de "pergunta -> resposta".
# E exatamente assim que um modelo base vira um "ChatGPT".
# =============================================================

# 1) CARREGA O MODELO BASE DO PASSO 4
salvo = torch.load("mini_gpt.pt")
chars = salvo["chars"]
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for c, i in ctoi.items()}
V = len(chars)
CONTEXTO, EMB, CABECAS, CAMADAS = 64, 128, 4, 4

def codificar(texto):
    # caracteres que o modelo nunca viu viram espaco
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

# 2) OS EXEMPLOS DE CONVERSA (o "material de estudo" do assistente)
conversas = []

saudacoes = ["Oi", "Ola", "Bom dia", "Boa tarde", "Boa noite", "E ai"]
for s in saudacoes:
    conversas.append((f"{s}!", "Olá! Em que posso ajudar?"))
    conversas.append((f"{s}, tudo bem?", "Tudo ótimo! E com você?"))

conversas += [
    ("Quem é você?", "Sou o mini-GPT, uma IA criada pelo Octavio."),
    ("Qual é o seu nome?", "Meu nome é mini-GPT."),
    ("Quem te criou?", "Fui criado pelo Octavio, do zero, em Python."),
    ("Você é uma IA?", "Sim, sou uma pequena IA treinada pelo Octavio."),
    ("O que você sabe fazer?", "Sei conversar um pouco e escrever como Machado."),
    ("Obrigado!", "De nada! Precisando, é só chamar."),
    ("Tchau!", "Tchau! Até a próxima."),
    ("Quem escreveu Dom Casmurro?", "Dom Casmurro foi escrito por Machado de Assis."),
    ("Quem é Capitu?", "Capitu é a personagem principal de Dom Casmurro."),
    ("Quem é Bentinho?", "Bentinho é o narrador de Dom Casmurro."),
    ("Quem foi Machado de Assis?", "Foi um grande escritor brasileiro."),
]

capitais = {
    "do Brasil": "Brasília", "de Portugal": "Lisboa", "da França": "Paris",
    "da Espanha": "Madri", "da Itália": "Roma", "da Alemanha": "Berlim",
    "do Japão": "Tóquio", "da China": "Pequim", "da Argentina": "Buenos Aires",
    "do Chile": "Santiago", "do Peru": "Lima", "do México": "Cidade do México",
    "do Canadá": "Otava", "da Rússia": "Moscou", "do Egito": "Cairo",
    "da Inglaterra": "Londres", "da Grécia": "Atenas", "do Uruguai": "Montevidéu",
    "da Colômbia": "Bogotá", "da Venezuela": "Caracas", "de Cuba": "Havana",
    "da Irlanda": "Dublin", "da Suécia": "Estocolmo", "da Noruega": "Oslo",
    "da Áustria": "Viena", "da Polônia": "Varsóvia", "da Bolívia": "La Paz",
    "do Paraguai": "Assunção", "do Equador": "Quito", "da Índia": "Nova Délhi",
}
# Alguns paises ficam FORA do treino, para testar depois
fora_do_treino = ["da Grécia", "da Noruega", "do Equador"]
for pais, capital in capitais.items():
    if pais not in fora_do_treino:
        conversas.append((f"Qual é a capital {pais}?",
                          f"A capital {pais} é {capital}."))

for a in range(1, 10):
    for b in range(1, 10):
        if (a + b) % 4 != 0:          # algumas contas ficam fora do treino
            conversas.append((f"Quanto é {a} mais {b}?",
                              f"{a} mais {b} é {a + b}."))

def formatar(pergunta, resposta):
    return f"Você: {pergunta}\nIA: {resposta}\n\n"

print(f"{len(conversas)} exemplos de conversa para o fine-tuning\n")

# Monta um "texto" com as conversas embaralhadas varias vezes
texto_conversa = ""
for _ in range(30):
    random.shuffle(conversas)
    texto_conversa += "".join(formatar(p, r) for p, r in conversas)
dados_conversa = torch.tensor(codificar(texto_conversa))

# Um pouco de Machado tambem, para ele nao "esquecer" o portugues
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

# 3) FINE-TUNING (so na primeira vez; depois carrega o arquivo salvo)
if os.path.exists("mini_gpt_chat.pt"):
    modelo.load_state_dict(torch.load("mini_gpt_chat.pt"))
    print("Modelo de conversa carregado (mini_gpt_chat.pt)\n")
else:
    modelo.load_state_dict(salvo["modelo"])
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=5e-4)
    PASSOS = 1000
    inicio = time.time()
    print("Ensinando o mini-GPT a conversar...")
    for passo in range(PASSOS + 1):
        # 80% das vezes estuda conversa, 20% estuda Machado
        fonte = dados_conversa if random.random() < 0.8 else dados_machado
        x, y = pegar_lote(fonte)
        perda = F.cross_entropy(modelo(x).view(-1, V), y.view(-1))
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
        if passo % 100 == 0:
            with torch.no_grad():
                xc, yc = pegar_lote(dados_conversa, 128)
                erro = F.cross_entropy(modelo(xc).view(-1, V), yc.view(-1))
            print(f"passo {passo:4d} | erro nas conversas: {erro.item():.3f}"
                  f" | {time.time() - inicio:.0f}s")
    torch.save(modelo.state_dict(), "mini_gpt_chat.pt")
    print("\nModelo de conversa salvo em mini_gpt_chat.pt\n")

modelo.eval()

# 4) CONVERSAR
@torch.no_grad()
def responder(pergunta, temperatura):
    ids = torch.tensor([codificar(f"Você: {pergunta}\nIA:")])
    resposta = ""
    for _ in range(150):
        logits = modelo(ids[:, -CONTEXTO:])[:, -1, :] / temperatura
        prox = torch.multinomial(F.softmax(logits, dim=-1), 1)
        letra = itoc[prox.item()]
        if letra == "\n":              # fim da resposta
            break
        resposta += letra
        print(letra, end="", flush=True)
        ids = torch.cat([ids, prox], dim=1)
    print()
    return resposta

temperatura = 0.5
print("=" * 60)
print(" Converse com o seu mini-GPT!")
print(" Comandos:  /temp 0.8  (muda a criatividade)   /sair")
print("=" * 60)
while True:
    try:
        pergunta = input("\nVoce: ").strip()
    except EOFError:
        break
    if pergunta == "/sair":
        break
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
