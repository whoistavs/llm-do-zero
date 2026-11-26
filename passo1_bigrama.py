import torch
import torch.nn.functional as F

torch.manual_seed(42)  # para os resultados serem sempre iguais

# 1) DADOS: a "experiência" da IA
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

# Cada letra vira um numero. O "." marca inicio e fim de um nome.
letras = sorted(set("".join(nomes)))
ltoi = {l: i + 1 for i, l in enumerate(letras)}
ltoi["."] = 0
itol = {i: l for l, i in ltoi.items()}
V = len(ltoi)

# Monta os pares (letra atual -> proxima letra)
xs, ys = [], []
for nome in nomes:
    seq = ["."] + list(nome) + ["."]
    for a, b in zip(seq, seq[1:]):
        xs.append(ltoi[a])
        ys.append(ltoi[b])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(f"{len(nomes)} nomes, {len(xs)} pares de letras para aprender\n")

# 2) MODELO: uma tabela de pesos que a IA vai ajustar
W = torch.randn((V, V), requires_grad=True)

def gerar(n=8):
    nomes_gerados = []
    for _ in range(n):
        i, saida = 0, []
        while True:
            probs = F.softmax(W[i], dim=0)
            i = torch.multinomial(probs, 1).item()
            if i == 0:
                break
            saida.append(itol[i])
        nomes_gerados.append("".join(saida).capitalize())
    return nomes_gerados

print("ANTES de treinar (chute aleatorio):")
print("  ", ", ".join(gerar()), "\n")

# 3) TREINO: errar, medir o erro, ajustar, repetir
taxa_aprendizado = 20.0
for passo in range(301):
    logits = W[xs]
    perda = F.cross_entropy(logits, ys)
    W.grad = None
    perda.backward()
    W.data -= taxa_aprendizado * W.grad
    if passo % 50 == 0:
        print(f"passo {passo:3d} | perda (erro): {perda.item():.3f}")

# 4) RESULTADO
print("\nDEPOIS de treinar (nomes inventados pela IA):")
print("  ", ", ".join(gerar(12)))