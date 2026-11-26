import torch
import torch.nn.functional as F

torch.manual_seed(42)

# =============================================================
# PASSO 3 - Aprender a escrever lendo LIVROS inteiros
#           (os 10 romances de Machado de Assis)
# =============================================================

# 1) DADOS: os 10 romances de Machado de Assis (dominio publico),
#    que vem junto com a biblioteca NLTK
import nltk
import zipfile
nltk.download("machado", quiet=True)
zip_path = str(nltk.data.find("corpora/machado.zip"))
texto = ""
livros = 0
with zipfile.ZipFile(zip_path) as z:
    for nome in sorted(z.namelist()):
        if "/romance/" in nome and nome.endswith(".txt"):
            texto += z.read(nome).decode("cp1252", errors="replace") + "\n\n"
            livros += 1
texto = texto.replace("\r", "")
print(f"Li {livros} livros do Machado de Assis")

# NOVIDADE: agora o vocabulario sao TODOS os caracteres dos livros
# (letras, maiusculas, acentos, pontuacao, espacos, quebras de linha)
chars = sorted(set(texto))
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for c, i in ctoi.items()}
V = len(chars)
dados = torch.tensor([ctoi[c] for c in texto])
print(f"Ao todo sao {len(texto):,} caracteres e {V} simbolos diferentes\n")

# NOVIDADE: separar dados de TREINO e de VALIDACAO
# A IA so estuda os primeiros 90%. Os ultimos 10% sao uma "prova"
# com texto que ela nunca viu: se ela for bem na prova, aprendeu
# de verdade; se for bem so no treino, apenas decorou.
corte = int(0.9 * len(dados))
treino, validacao = dados[:corte], dados[corte:]

CONTEXTO = 8   # olha os ultimos 8 caracteres

def pegar_lote(fonte, tamanho):
    pos = torch.randint(0, len(fonte) - CONTEXTO, (tamanho,))
    x = torch.stack([fonte[p:p + CONTEXTO] for p in pos])
    y = fonte[pos + CONTEXTO]
    return x, y

# 2) MODELO: mesma ideia do passo 2, so que maior
EMB = 24
OCULTA = 256
C  = torch.randn((V, EMB))
W1 = torch.randn((CONTEXTO * EMB, OCULTA)) * (1 / (CONTEXTO * EMB) ** 0.5)
b1 = torch.zeros(OCULTA)
W2 = torch.randn((OCULTA, V)) * 0.01
b2 = torch.zeros(V)
parametros = [C, W1, b1, W2, b2]
for p in parametros:
    p.requires_grad = True
print(f"A rede tem {sum(p.nelement() for p in parametros):,} parametros\n")

def prever(x):
    h = torch.tanh(C[x].view(x.shape[0], -1) @ W1 + b1)
    return h @ W2 + b2

@torch.no_grad()
def medir(fonte):
    x, y = pegar_lote(fonte, 20000)
    return F.cross_entropy(prever(x), y).item()

@torch.no_grad()
def escrever(comeco="Capitu ", tamanho=300):
    janela = [ctoi.get(c, 0) for c in comeco[-CONTEXTO:].rjust(CONTEXTO)]
    saida = comeco
    for _ in range(tamanho):
        probs = F.softmax(prever(torch.tensor([janela])), dim=1)
        i = torch.multinomial(probs, 1).item()
        saida += itoc[i]
        janela = janela[1:] + [i]
    return saida

print("ANTES de treinar:")
print(escrever(), "\n")

# 3) TREINO
PASSOS = 20000
for passo in range(PASSOS + 1):
    x, y = pegar_lote(treino, 64)
    perda = F.cross_entropy(prever(x), y)
    for p in parametros:
        p.grad = None
    perda.backward()
    taxa = 0.1 if passo < PASSOS * 0.75 else 0.01
    for p in parametros:
        p.data -= taxa * p.grad
    if passo % 2000 == 0:
        print(f"passo {passo:5d} | erro no treino: {medir(treino):.3f}"
              f" | erro na prova: {medir(validacao):.3f}")

# 4) RESULTADO
print("\nDEPOIS de treinar - a IA escrevendo no estilo de Machado:\n")
print(escrever())