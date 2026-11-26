import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta

import jaci as t

# =============================================================
# AVALIAR - faz uma bateria de perguntas à Jaci e da uma nota
# -------------------------------------------------------------
# Cada teste tem uma pergunta e regras simples para conferir a
# resposta. Os testes estao divididos em:
#   treino -> usados para ajustar a Jaci
#   prova  -> NAO usados para ajustar; mostram se ele melhorou de
#             verdade ou so "decorou" o treino (overfitting)
#
# Uso:
#   python avaliar.py           roda todos os testes
#   python avaliar.py treino    roda so os de treino
#   python avaliar.py prova     roda so os de prova
# =============================================================

DIAS = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
hoje = datetime.now()
NAO_SEI = ["nao sei", "nao achei", "nao localizei", "nenhuma informacao", "nao mencion",
           "nao inclu", "nao contem", "nao constam", "nao aparece", "nao tenho", "nao encontr", "nao consegui", "nao ha", "nao esta",
           "nao consta", "nao especific", "nao possuo", "nao tenho acesso", "nao informa",
           "a ser agendad", "posteriormente", "nao foi definid", "nao foi especific"]

# Como ler as regras:
#   "deve": lista de grupos; em cada grupo, pelo menos UMA opcao precisa aparecer
#   "nao_deve": nenhuma dessas pode aparecer
#   "ferramenta": a ferramenta que ele deveria usar (ou None se tanto faz)
TESTES = [
    # ---------------- TREINO ----------------
    {"id": "hora-1", "grupo": "treino", "categoria": "hora",
     "pergunta": "Que dia da semana é hoje?",
     "deve": [[DIAS[hoje.weekday()]]], "ferramenta": None},
    {"id": "conta-1", "grupo": "treino", "categoria": "conta",
     "pergunta": "Quanto é 1847 vezes 23?",
     "deve": [["42481"]], "nao_deve": ["$", "\\times"], "ferramenta": "calcular"},
    {"id": "conta-2", "grupo": "treino", "categoria": "conta",
     "pergunta": "Se eu gastar 37,50 euros por semana, quanto gasto em 12 semanas?",
     "deve": [["450"]], "ferramenta": "calcular"},
    {"id": "conta-3", "grupo": "treino", "categoria": "conta",
     "pergunta": "Quanto é 2 elevado a 10?",
     "deve": [["1024"]], "nao_deve": ["$", "\\"], "ferramenta": "calcular"},
    {"id": "clima-1", "grupo": "treino", "categoria": "clima",
     "pergunta": "Vai chover amanhã em Barcelos?",
     "deve": [["chuva", "chover", "precipita"]], "ferramenta": "clima"},
    {"id": "wiki-1", "grupo": "treino", "categoria": "wikipedia",
     "pergunta": "Quais times brasileiros foram campeões do Mundial de Clubes da FIFA?",
     "deve": [["corinthians"], ["sao paulo"], ["internacional"], ["2012"]],
     "ferramenta": "buscar_wikipedia"},
    {"id": "wiki-2", "grupo": "treino", "categoria": "wikipedia",
     "pergunta": "Em que ano nasceu Machado de Assis?",
     "deve": [["1839"]], "ferramenta": "buscar_wikipedia"},
    {"id": "wiki-3", "grupo": "treino", "categoria": "wikipedia",
     "pergunta": "Quem é o autor de O Guardador de Rebanhos?",
     "deve": [["caeiro", "pessoa"]], "ferramenta": "buscar_wikipedia"},
    {"id": "docs-1", "grupo": "treino", "categoria": "documentos",
     "pergunta": "Qual é o número máximo de alunos por grupo no projeto 2 de IA?",
     "deve": [["cinco", r"re:(?<![\d.,])5(?![\d])"]], "ferramenta": "buscar_documentos"},
    {"id": "docs-2", "grupo": "treino", "categoria": "documentos",
     "pergunta": "O que o Project Guide 02 de IA pede para entregar no arquivo ZIP?",
     "deve": [["notebook", "jupyter"], ["readme"]], "ferramenta": "buscar_documentos"},
    {"id": "honesto-1", "grupo": "treino", "categoria": "honestidade",
     "pergunta": "Qual é o prazo de entrega do Project Guide 02 de IA?",
     "deve": [NAO_SEI], "ferramenta": "buscar_documentos"},
    {"id": "memoria-1", "grupo": "treino", "categoria": "memoria",
     "pergunta": "Onde eu moro?",
     "deve": [["barcelos"]], "ferramenta": None},
    {"id": "idioma-1", "grupo": "treino", "categoria": "pt-br",
     "pergunta": "Me dá uma ideia de café da manhã rápido.",
     "nao_deve": ["pequeno-almoco", "pequeno almoco", "estou a ", "esta a fazer"],
     "ferramenta": None},

    # ---------------- PROVA (nao ajustar em cima destes!) ----------------
    {"id": "conta-p1", "grupo": "prova", "categoria": "conta",
     "pergunta": "Quanto é 15% de 2340?",
     "deve": [["351"]], "ferramenta": "calcular"},
    {"id": "wiki-p1", "grupo": "prova", "categoria": "wikipedia",
     "pergunta": "Quem pintou a Mona Lisa?",
     "deve": [["leonardo", "da vinci"]], "ferramenta": None},
    {"id": "wiki-p2", "grupo": "prova", "categoria": "wikipedia",
     "pergunta": "Em que ano o Brasil ganhou sua primeira Copa do Mundo de futebol?",
     "deve": [["1958"]], "ferramenta": "buscar_wikipedia"},
    {"id": "wiki-p3", "grupo": "prova", "categoria": "wikipedia",
     "pergunta": "Qual é a capital da Austrália?",
     "deve": [["camberra", "canberra"]], "ferramenta": None},
    {"id": "hora-p1", "grupo": "prova", "categoria": "hora",
     "pergunta": "Que dia da semana vai ser daqui a 3 dias?",
     "deve": [[DIAS[(hoje + timedelta(days=3)).weekday()]]], "ferramenta": None},
    {"id": "docs-p1", "grupo": "prova", "categoria": "documentos",
     "pergunta": "Que nome o repositório do GitHub do projeto 2 de IA deve ter?",
     "deve": [["ia26_p02"]], "ferramenta": "buscar_documentos"},
    {"id": "docs-p2", "grupo": "prova", "categoria": "documentos",
     "pergunta": "Qual algoritmo o projeto 2 de IA pede para as regras de associação?",
     "deve": [["apriori"]], "ferramenta": "buscar_documentos"},
    {"id": "wiki-p4", "grupo": "prova", "categoria": "wikipedia",
     "pergunta": "Quem escreveu Memórias Póstumas de Brás Cubas e em que ano foi publicado?",
     "deve": [["machado"], ["1881"]], "ferramenta": None},
    {"id": "honesto-p2", "grupo": "prova", "categoria": "honestidade",
     "pergunta": "Qual é o nome do meu professor de Inteligência Artificial?",
     "deve": [NAO_SEI], "ferramenta": None},
    {"id": "honesto-p1", "grupo": "prova", "categoria": "honestidade",
     "pergunta": "Que nota eu tirei na última prova de Redes?",
     "deve": [NAO_SEI], "ferramenta": None},
]


PALAVRAS_PT_PT = ["heterónimo", "pequeno-almoço", "telemóvel", "autocarro", "ecrã", "facto "]


# Testes criados a partir dos erros reportados com o botao 👎 na interface
# (o Claude transforma os erros de erros_reportados.json em testes aqui)
ARQ_EXTRAS = "testes_extras.json"
try:
    with open(ARQ_EXTRAS, encoding="utf-8") as f:
        TESTES += json.load(f)
except FileNotFoundError:
    pass


def normalizar(texto):
    """minusculas, sem acentos, sem markdown e sem separador de milhar (42.481 -> 42481)"""
    texto = re.sub(r"[*_`#]", "", texto)
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"(?<=\d)[.\s](?=\d{3}\b)", "", texto)


def conferir(teste, resposta, ferramentas):
    problemas = []
    r = normalizar(resposta)
    for grupo in teste.get("deve", []):
        if not any(re.search(op[3:], r) if op.startswith("re:") else normalizar(op) in r
                   for op in grupo):
            problemas.append("faltou: " + " ou ".join(o.replace("re:", "") for o in grupo[:4]))
    for proibido in teste.get("nao_deve", []):
        if normalizar(proibido) in r:
            problemas.append(f"nao devia conter: {proibido}")
    for palavra in PALAVRAS_PT_PT:
        if palavra in resposta.lower():
            problemas.append(f"portugues de Portugal: {palavra.strip()}")
    esperada = teste.get("ferramenta")
    if esperada and esperada not in ferramentas:
        problemas.append(f"nao usou a ferramenta {esperada}")
    return problemas


def rodar(teste, memoria):
    # "antes": conversa anterior (para testar erros que so aparecem no meio de um papo)
    mensagens = [{"role": "system", "content": t.montar_sistema(memoria)}]
    mensagens += t.historico_para_mensagens(teste.get("antes", []))
    mensagens.append({"role": "user", "content": teste["pergunta"]})
    resposta, ferramentas = "", []
    inicio = time.time()
    try:
        for parte in t.responder_em_partes(mensagens):
            if parte["tipo"] == "texto":
                resposta += parte["conteudo"]
            else:
                args = ", ".join(f"{k}={v!r}" for k, v in parte["argumentos"].items())
                ferramentas.append(f"{parte['nome']}({args})")
                resposta = ""
    except Exception as erro:
        resposta = f"[ERRO: {erro}]"
    return resposta.strip(), ferramentas, time.time() - inicio


def main():
    grupo = sys.argv[1] if len(sys.argv) > 1 else None
    testes = [x for x in TESTES if grupo is None or x["grupo"] == grupo]
    memoria = t.carregar(t.ARQ_MEMORIA, [])
    print("Conferindo a pasta documentos...")
    t.indexar(mostrar=False)
    print(f"Rodando {len(testes)} testes (pode levar alguns minutos)...\n")

    resultados = []
    for i, teste in enumerate(testes, 1):
        print(f"[{i}/{len(testes)}] {teste['pergunta']}", flush=True)
        resposta, ferramentas, segundos = rodar(teste, memoria)
        problemas = conferir(teste, resposta, " ".join(ferramentas))
        status = "PASSOU" if not problemas else "FALHOU"
        print(f"    {status} ({segundos:.0f}s)" + (f" - {'; '.join(problemas)}" if problemas else ""))
        resultados.append({**teste, "resposta": resposta, "ferramentas_usadas": ferramentas,
                           "segundos": round(segundos, 1), "problemas": problemas,
                           "passou": not problemas})

    # ---------- resumo ----------
    linhas = [f"# Relatório de avaliação da Jaci",
              f"Data: {hoje.strftime('%d/%m/%Y %H:%M')} | Modelo: {t.MODELO}", ""]
    for g in ("treino", "prova"):
        rs = [r for r in resultados if r["grupo"] == g]
        if rs:
            ok = sum(r["passou"] for r in rs)
            linhas.append(f"**{g.capitalize()}: {ok}/{len(rs)} ({100 * ok // len(rs)}%)**  ")
    cats = sorted({r["categoria"] for r in resultados})
    linhas += ["", "| Categoria | Acertos |", "|---|---|"]
    for c in cats:
        rs = [r for r in resultados if r["categoria"] == c]
        linhas.append(f"| {c} | {sum(r['passou'] for r in rs)}/{len(rs)} |")
    linhas += ["", "## Detalhes", ""]
    for r in resultados:
        linhas += [f"### {'✅' if r['passou'] else '❌'} {r['id']} ({r['grupo']}, {r['segundos']}s)",
                   f"**Pergunta:** {r['pergunta']}  ",
                   f"**Ferramentas:** {', '.join(r['ferramentas_usadas']) or 'nenhuma'}  "]
        if r["problemas"]:
            linhas.append(f"**Problemas:** {'; '.join(r['problemas'])}  ")
        linhas += ["**Resposta:**", "", "> " + r["resposta"].replace("\n", "\n> "), ""]

    with open("relatorio_avaliacao.md", "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    with open("relatorio_avaliacao.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\n" + "\n".join(linhas[2:4 + 1]))
    print("\nRelatório completo salvo em relatorio_avaliacao.md")


if __name__ == "__main__":
    main()
