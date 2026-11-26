import time
import zipfile
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# =============================================================
# PASSO 4 - Um mini-GPT (Transformer) treinado com Machado de Assis
# =============================================================

# 1) DADOS (os mesmos 10 romances do passo 3)
nltk.download("machado", quiet=True)
zip_path = str(nltk.data.find("corpora/machado.zip"))
texto = ""
with zipfile.ZipFile(zip_path) as z:
    for nome in sorted(z.namelist()):
        if "/romance/" in nome and nome.endswith(".txt"):
            texto += z.read(nome).decode("cp1252", errors="replace") + "\n\n"
texto = texto.replace("\r", "")

chars = sorted(set(texto))
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for c, i in ctoi.items()}
V = len(chars)
dados = torch.tensor([ctoi[c] for c in texto])
corte = int(0.9 * len(dados))
treino, validacao = dados[:corte], dados[corte:]

# 2) CONFIGURACAO DO MODELO
CONTEXTO = 64     # agora olha os ultimos 64 caracteres (antes eram 8)
EMB = 128         # tamanho do vetor de cada caractere
CABECAS = 4       # quantas "atencoes" em paralelo
CAMADAS = 4       # quantos blocos Transformer empilhados
LOTE = 32
PASSOS = 3000

def pegar_lote(fonte):
    pos = torch.randint(0, len(fonte) - CONTEXTO - 1, (LOTE,))
    x = torch.stack([fonte[p:p + CONTEXTO] for p in pos])
    y = torch.stack([fonte[p + 1:p + CONTEXTO + 1] for p in pos])
    return x, y

# 3) O TRANSFORMER
# NOVIDADE: ATENCAO. Cada caractere "olha" para todos os anteriores e
# decide quais sao importantes para prever o proximo. Ex: para fechar
# um dialogo, a IA precisa lembrar que um travessao abriu la atras.
class Atencao(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMB, 3 * EMB)   # pergunta, chave e valor
        self.saida = nn.Linear(EMB, EMB)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(EMB, dim=2)
        q = q.view(B, T, CABECAS, -1).transpose(1, 2)
        k = k.view(B, T, CABECAS, -1).transpose(1, 2)
        v = v.view(B, T, CABECAS, -1).transpose(1, 2)
        # is_causal=True: so pode olhar para TRAS, nunca para o futuro
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.saida(y.transpose(1, 2).reshape(B, T, EMB))

class Bloco(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMB)
        self.atencao = Atencao()               # 1) se comunicar
        self.norm2 = nn.LayerNorm(EMB)
        self.pensar = nn.Sequential(           # 2) pensar sozinho
            nn.Linear(EMB, 4 * EMB), nn.GELU(), nn.Linear(4 * EMB, EMB))

    def forward(self, x):
        x = x + self.atencao(self.norm1(x))
        x = x + self.pensar(self.norm2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_char = nn.Embedding(V, EMB)          # QUAL caractere
        self.emb_pos = nn.Embedding(CONTEXTO, EMB)    # ONDE ele esta
        self.blocos = nn.Sequential(*[Bloco() for _ in range(CAMADAS)])
        self.norm = nn.LayerNorm(EMB)
        self.cabeca = nn.Linear(EMB, V)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb_char(x) + self.emb_pos(torch.arange(T))
        return self.cabeca(self.norm(self.blocos(h)))

    @torch.no_grad()
    def escrever(self, comeco, tamanho=500, temperatura=0.8):
        ids = torch.tensor([[ctoi.get(c, 0) for c in comeco]])
        for _ in range(tamanho):
            logits = self(ids[:, -CONTEXTO:])[:, -1, :] / temperatura
            prox = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, prox], dim=1)
        return "".join(itoc[i] for i in ids[0].tolist())

modelo = MiniGPT()
print(f"Mini-GPT com {sum(p.numel() for p in modelo.parameters()):,} parametros")
print(f"Treinando com {len(texto):,} caracteres de Machado de Assis\n")

@torch.no_grad()
def medir(fonte):
    modelo.eval()
    erros = []
    for _ in range(20):
        x, y = pegar_lote(fonte)
        erros.append(F.cross_entropy(modelo(x).view(-1, V), y.view(-1)).item())
    modelo.train()
    return sum(erros) / len(erros)

# 4) TREINO (otimizador AdamW, o mesmo usado nos GPTs de verdade)
otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3)
inicio = time.time()
for passo in range(PASSOS + 1):
    x, y = pegar_lote(treino)
    perda = F.cross_entropy(modelo(x).view(-1, V), y.view(-1))
    otimizador.zero_grad()
    perda.backward()
    otimizador.step()
    if passo % 250 == 0:
        print(f"passo {passo:4d} | treino: {medir(treino):.3f}"
              f" | prova: {medir(validacao):.3f}"
              f" | {time.time() - inicio:.0f}s")

# 5) SALVA O CEREBRO TREINADO (para usar depois sem treinar de novo)
torch.save({"modelo": modelo.state_dict(), "chars": chars}, "mini_gpt.pt")
print("\nModelo salvo em mini_gpt.pt")

# 6) RESULTADO
modelo.eval()
print("\nO mini-GPT escrevendo como Machado de Assis:\n")
print(modelo.escrever("Capitu olhou para mim e disse"))