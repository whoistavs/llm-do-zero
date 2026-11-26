import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================
# PASSO 5 - Usar o mini-GPT treinado (sem treinar de novo)
# =============================================================

# 1) CARREGA O CEREBRO SALVO NO PASSO 4
salvo = torch.load("mini_gpt.pt")
chars = salvo["chars"]
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for c, i in ctoi.items()}
V = len(chars)

# A configuracao precisa ser IGUAL a do treino
CONTEXTO, EMB, CABECAS, CAMADAS = 64, 128, 4, 4

# 2) A MESMA ARQUITETURA DO PASSO 4
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
modelo.load_state_dict(salvo["modelo"])
modelo.eval()

# 3) ESCREVE NA TELA LETRA POR LETRA (igual ao ChatGPT)
@torch.no_grad()
def escrever(comeco, tamanho, temperatura):
    ids = torch.tensor([[ctoi.get(c, ctoi[" "]) for c in comeco]])
    for _ in range(tamanho):
        logits = modelo(ids[:, -CONTEXTO:])[:, -1, :] / temperatura
        prox = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, prox], dim=1)
        print(itoc[prox.item()], end="", flush=True)
    print("\n")

# 4) LOOP: voce escreve o comeco, o mini-GPT continua
temperatura = 0.8
print("=" * 60)
print(" Seu mini-GPT esta pronto!")
print(" Escreva o COMECO de um texto e ele continua.")
print(" Comandos:  /temp 0.5  (muda a criatividade)   /sair")
print("=" * 60)
while True:
    try:
        comeco = input("\nVoce: ")
    except EOFError:
        break
    if comeco.strip() == "/sair":
        break
    if comeco.startswith("/temp"):
        try:
            temperatura = float(comeco.split()[1])
            print(f"Temperatura agora e {temperatura}")
        except (IndexError, ValueError):
            print("Use assim: /temp 0.5")
        continue
    if not comeco.strip():
        continue
    print("\nMini-GPT: " + comeco, end="", flush=True)
    escrever(comeco, 300, temperatura)