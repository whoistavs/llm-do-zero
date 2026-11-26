import torch
import torch.nn.functional as F

torch.manual_seed(42)

# =============================================================
# PASSO 2 - Rede neural que olha as ULTIMAS 3 LETRAS
# =============================================================

# 1) DADOS (mesma lista do passo 1)
nomes = """
ana maria joao pedro lucas gabriel rafael mateus gustavo felipe bruno
carlos daniel eduardo fernando henrique igor julio leonardo marcelo
nicolas otavio paulo ricardo rodrigo samuel thiago vinicius vitor wagner
beatriz camila carolina clara debora eduarda fernanda gabriela helena
isabela julia larissa leticia luana mariana natalia patricia rafaela
renata sabrina tatiana vanessa yasmin alice bianca cecilia elisa giovana
heloisa jessica lara livia manuela melissa olivia priscila raquel sofia
valentina antonio arthur benicio caio davi enzo erick fabio heitor
joaquim kaua lorenzo miguel murilo noah otto raul sergio theo yuri
adriana aline amanda andreia barbara cristina daniela denise elaine
flavia graziela ingrid jaqueline juliana karina lorena luciana marcia
monica nathalia paula regina sandra simone tais viviane alessandro
andre augusto cesar cristiano diego emerson fabricio francisco geraldo
guilherme hugo jefferson jorge leandro luiz mauricio mauro nelson
osvaldo renan roberto rogerio sandro silvio tiago valter wellington
""".split()

letras = sorted(set("".join(nomes)))
ltoi = {l: i + 1 for i, l in enumerate(letras)}
ltoi["."] = 0
itol = {i: l for l, i in ltoi.items()}
V = len(ltoi)

# NOVIDADE: o "contexto" agora sao as 3 ultimas letras, nao so 1
CONTEXTO = 3

xs, ys = [], []
for nome in nomes:
    janela = [0] * CONTEXTO              # comeca com "..."
    for letra in nome + ".":
        xs.append(janela)                # ex: ['.','.','a'] -> 'n'
        ys.append(ltoi[letra])
        janela = janela[1:] + [ltoi[letra]]
xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(f"{len(xs)} exemplos de treino (3 letras -> proxima letra)\n")

# 2) MODELO: uma rede neural de verdade, com 3 partes
EMB = 10      # cada letra vira um vetor de 10 numeros (embedding)
OCULTA = 100  # neuronios na camada escondida

C  = torch.randn((V, EMB))                        # tabela de embeddings
W1 = torch.randn((CONTEXTO * EMB, OCULTA)) * 0.2  # camada escondida
b1 = torch.zeros(OCULTA)
W2 = torch.randn((OCULTA, V)) * 0.1               # camada de saida
b2 = torch.zeros(V)
parametros = [C, W1, b1, W2, b2]
for p in parametros:
    p.requires_grad = True
print(f"A rede tem {sum(p.nelement() for p in parametros)} parametros\n")

def prever(x):
    emb = C[x]                                # letras -> vetores
    h = torch.tanh(emb.view(x.shape[0], -1) @ W1 + b1)  # "pensa"
    return h @ W2 + b2                        # pontuacao de cada letra

def gerar(n=8):
    resultado = []
    for _ in range(n):
        janela, saida = [0] * CONTEXTO, []
        while True:
            probs = F.softmax(prever(torch.tensor([janela])), dim=1)
            i = torch.multinomial(probs, 1).item()
            if i == 0:
                break
            saida.append(itol[i])
            janela = janela[1:] + [i]
        resultado.append("".join(saida).capitalize())
    return resultado

print("ANTES de treinar:")
print("  ", ", ".join(gerar()), "\n")

# 3) TREINO (agora em "lotes": 32 exemplos sorteados por vez)
for passo in range(5001):
    lote = torch.randint(0, xs.shape[0], (32,))
    perda = F.cross_entropy(prever(xs[lote]), ys[lote])
    for p in parametros:
        p.grad = None
    perda.backward()
    taxa = 0.1 if passo < 3000 else 0.01      # passos menores no final
    for p in parametros:
        p.data -= taxa * p.grad
    if passo % 1000 == 0:
        with torch.no_grad():
            perda_total = F.cross_entropy(prever(xs), ys).item()
        print(f"passo {passo:4d} | perda (erro): {perda_total:.3f}")

# 4) RESULTADO
print("\nDEPOIS de treinar:")
print("  ", ", ".join(gerar(12)))