import ast
import glob
import json
import math
import operator
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser

import ollama

# =============================================================
# JACI - assistente pessoal com MEMORIA, FERRAMENTAS e SEUS ARQUIVOS
# -------------------------------------------------------------
# Usa o modelo "jaci" do Ollama e guarda duas coisas no disco:
#   memoria.json   -> fatos sobre voce que ele deve sempre lembrar
#   historico.json -> as ultimas mensagens, para continuar a conversa
# =============================================================

MODELO = "jaci"
ARQ_MEMORIA = "memoria.json"
ARQ_HISTORICO = "historico.json"
MAX_HISTORICO = 20   # quantas mensagens antigas ele relembra ao abrir

REGRAS = """Você é a Jaci, a assistente pessoal do Octavio.
Regras:
- Responda sempre em português do Brasil, nunca em português de Portugal.
- Seja direto: respostas curtas para perguntas simples, longas só quando ele pedir detalhes.
- Use listas numeradas quando explicar um passo a passo.
- Se não tiver certeza de um fato, citação ou data, diga que não tem certeza em vez de inventar.
- Use o que você sabe sobre o Octavio só quando ajudar na resposta.
- Você tem ferramentas. Use-as em vez de chutar: clima para o tempo,
  hora_atual para data e hora, calcular para qualquer conta e
  buscar_wikipedia para conferir fatos, livros, pessoas e datas e
  buscar_documentos para perguntas sobre os arquivos, trabalhos e aulas do Octavio.
- Para analisar dados, fazer gráficos, testar algoritmos ou contas com muitos
  passos, use executar_python. O Octavio precisa aprovar cada execução.
  Salve arquivos só com o nome (na pasta area_python), nunca em ../documentos.
- NUNCA diga que gerou, salvou, rodou, consultou ou calculou algo sem ter
  chamado a ferramenta NESTA resposta. Se o Octavio pedir para refazer um
  gráfico ou código, chame executar_python de novo com o código novo.
- Se a pergunta falar de um trabalho, projeto, guia, enunciado, matéria ou aula
  do Octavio, SEMPRE consulte buscar_documentos antes de responder, mesmo que
  pareça um pedido de sugestão. Só peça mais detalhes se os documentos não ajudarem.
- Ao responder com base nos documentos, diga de qual arquivo e página (ou slide) veio.
- Quando usar uma ferramenta, baseie a resposta SÓ no que ela retornou.
  Se o resultado não tiver a informação pedida, diga isso claramente
  em vez de completar com o que você acha que sabe.
- Escreva texto simples: não use LaTeX nem símbolos como $ ou \\times.
- Use sempre a grafia brasileira, mesmo que a fonte esteja em português de
  Portugal (ex: "heterônimo", não "heterónimo"; "fato", não "facto")."""


# =============================================================
# FERRAMENTAS: funcoes que a Jaci pode decidir usar sozinha.
# O modelo le o nome e a descricao (docstring) de cada uma e,
# quando precisa, pede para o programa rodar e devolver o resultado.
# =============================================================

def _baixar_json(url):
    pedido = urllib.request.Request(url, headers={"User-Agent": "Jaci/1.0 (projeto pessoal; github.com/whoistavs/llm-do-zero)"})
    with urllib.request.urlopen(pedido, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


CODIGOS_TEMPO = {
    0: "céu limpo", 1: "quase limpo", 2: "parcialmente nublado", 3: "nublado",
    45: "neblina", 48: "neblina", 51: "garoa fraca", 53: "garoa", 55: "garoa forte",
    61: "chuva fraca", 63: "chuva", 65: "chuva forte", 71: "neve fraca",
    73: "neve", 75: "neve forte", 80: "pancadas de chuva fracas",
    81: "pancadas de chuva", 82: "pancadas de chuva fortes", 95: "trovoada",
    96: "trovoada com granizo", 99: "trovoada com granizo",
}


def clima(cidade: str) -> str:
    """Consulta a previsão do tempo real para hoje e os próximos 2 dias.

    Args:
        cidade: Nome da cidade com o país depois de uma vírgula, ex: "Barcelos, Portugal".
    """
    nome, _, pais = cidade.partition(",")
    busca = urllib.parse.quote(nome.strip())
    dados = _baixar_json("https://geocoding-api.open-meteo.com/v1/search"
                         f"?name={busca}&count=10&language=pt")
    lugares = dados.get("results") or []
    if pais.strip():
        filtrados = [l for l in lugares
                     if pais.strip().lower() in l.get("country", "").lower()]
        lugares = filtrados or lugares
    if not lugares:
        return f"Não encontrei a cidade {cidade}."
    lugar = lugares[0]
    prev = _baixar_json(
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lugar['latitude']}&longitude={lugar['longitude']}"
        "&current=temperature_2m,weather_code"
        "&daily=temperature_2m_max,temperature_2m_min,"
        "precipitation_probability_max,weather_code"
        "&timezone=auto&forecast_days=3")
    agora = prev["current"]
    d = prev["daily"]
    linhas = [f"Local: {lugar['name']}, {lugar.get('admin1', '')}, {lugar.get('country', '')}",
              f"Agora: {agora['temperature_2m']}°C, "
              f"{CODIGOS_TEMPO.get(agora['weather_code'], 'tempo variado')}"]
    for i in range(len(d["time"])):
        linhas.append(
            f"{d['time'][i]}: mín {d['temperature_2m_min'][i]}°C, "
            f"máx {d['temperature_2m_max'][i]}°C, "
            f"chance de chuva {d['precipitation_probability_max'][i]}%, "
            f"{CODIGOS_TEMPO.get(d['weather_code'][i], 'tempo variado')}")
    return "\n".join(linhas)


def hora_atual() -> str:
    """Retorna a data, o dia da semana e a hora exatos de agora."""
    dias = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
            "sexta-feira", "sábado", "domingo"]
    agora = datetime.now()
    return f"{dias[agora.weekday()]}, {agora.strftime('%d/%m/%Y %H:%M')}"


_OPERACOES = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _avaliar(no):
    if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
        return no.value
    if isinstance(no, ast.BinOp) and type(no.op) in _OPERACOES:
        return _OPERACOES[type(no.op)](_avaliar(no.left), _avaliar(no.right))
    if isinstance(no, ast.UnaryOp) and type(no.op) in _OPERACOES:
        return _OPERACOES[type(no.op)](_avaliar(no.operand))
    raise ValueError("expressão não permitida")


def calcular(expressao: str) -> str:
    """Faz contas matemáticas com precisão. Use sempre que houver números.

    Args:
        expressao: Conta em formato Python, ex: "1250 * 0.23" ou "(15 + 7) / 3".
    """
    try:
        texto = expressao.replace(",", ".").replace("^", "**").replace("x", "*")
        resultado = _avaliar(ast.parse(texto, mode="eval").body)
        return f"{expressao} = {round(resultado, 6)}"
    except Exception as erro:
        return f"Não consegui calcular '{expressao}': {erro}"


class _LeitorHTML(HTMLParser):
    """Transforma o HTML de um artigo em texto simples, mantendo as tabelas
    (cada linha da tabela vira uma linha de texto com colunas separadas por |)."""
    IGNORAR = {"style", "script", "sup", "math"}
    QUEBRA = {"p", "li", "h2", "h3", "h4", "tr", "br", "div", "dd", "dt", "caption"}

    def __init__(self):
        super().__init__()
        self.partes, self.pulando = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNORAR:
            self.pulando += 1
        elif tag in ("td", "th"):
            self.partes.append(" | ")
        elif tag in ("h2", "h3"):
            self.partes.append("\n## ")

    def handle_endtag(self, tag):
        if tag in self.IGNORAR:
            self.pulando = max(0, self.pulando - 1)
        elif tag in self.QUEBRA:
            self.partes.append("\n")

    def handle_data(self, data):
        if not self.pulando:
            self.partes.append(data)

    def texto(self):
        linhas = (" ".join(l.split()) for l in "".join(self.partes).split("\n"))
        return "\n".join(l for l in linhas if l.strip(" |"))


_cache_wiki = {}
PASTA_CACHE_WIKI = "cache_wiki"


def _arquivo_cache(titulo):
    nome = re.sub(r"[^\w-]+", "_", titulo)[:80]
    return os.path.join(PASTA_CACHE_WIKI, nome + ".json")


def _salvar_cache_wiki(titulo):
    os.makedirs(PASTA_CACHE_WIKI, exist_ok=True)
    dados = _cache_wiki[titulo]
    salvar(_arquivo_cache(titulo), {"pedacos": dados["pedacos"],
                                    "vetores": {str(k): v for k, v in dados["vetores"].items()}})


def buscar_wikipedia(termo: str, pergunta: str = "") -> str:
    """Lê um artigo inteiro da Wikipédia em português (incluindo tabelas) e devolve os trechos mais relevantes. Use para conferir fatos, datas, listas, campeões, pessoas, livros e lugares.

    Args:
        termo: Nome do artigo a pesquisar, ex: "Copa do Mundo de Clubes da FIFA".
        pergunta: O que você quer descobrir no artigo, ex: "lista de campeões brasileiros".
    """
    busca = urllib.parse.quote(termo)
    achados = _baixar_json("https://pt.wikipedia.org/w/api.php?action=query&list=search"
                           f"&srsearch={busca}&srlimit=1&format=json")
    resultados = achados.get("query", {}).get("search", [])
    if not resultados:
        return f"Nada encontrado na Wikipédia sobre '{termo}'."
    titulo = resultados[0]["title"]

    if titulo not in _cache_wiki and os.path.exists(_arquivo_cache(titulo)):
        salvo = carregar(_arquivo_cache(titulo), {})
        _cache_wiki[titulo] = {"pedacos": salvo["pedacos"],
                               "vetores": {int(k): v for k, v in salvo["vetores"].items()}}
    if titulo not in _cache_wiki:
        pagina = _baixar_json("https://pt.wikipedia.org/w/api.php?action=parse&prop=text"
                              f"&page={urllib.parse.quote(titulo)}&redirects=1"
                              "&format=json&formatversion=2")
        leitor = _LeitorHTML()
        leitor.feed(pagina.get("parse", {}).get("text", ""))
        texto = leitor.texto()
        # divide em pedacos respeitando as linhas (para nao cortar tabelas no meio)
        pedacos, atual = [], ""
        for linha in texto.split("\n"):
            if len(atual) + len(linha) > TAM_PEDACO and atual:
                pedacos.append(atual)
                atual = ""
            atual += linha + "\n"
        if atual:
            pedacos.append(atual)
        _cache_wiki[titulo] = {"pedacos": pedacos[:600], "vetores": {}}
        _salvar_cache_wiki(titulo)

    cache = _cache_wiki[titulo]
    pedacos = cache["pedacos"]
    consulta_txt = pergunta or termo
    # Pre-filtro barato: so calcula embeddings dos 40 trechos que mais
    # compartilham palavras com a pergunta (muito mais rapido em artigos longos)
    palavras = {p for p in re.findall(r"\w+", (consulta_txt + " " + termo).lower()) if len(p) > 3}
    pontos = [(sum(p in trecho.lower() for p in palavras), -i) for i, trecho in enumerate(pedacos)]
    candidatos = sorted(range(len(pedacos)), key=lambda i: pontos[i], reverse=True)[:40]
    faltam = [i for i in candidatos if i not in cache["vetores"]]
    if faltam:
        vets = _embeddings([f"title: {titulo} | text: {pedacos[i]}" for i in faltam])
        cache["vetores"].update(zip(faltam, vets))
        _salvar_cache_wiki(titulo)
    consulta = _embeddings([f"task: search result | query: {consulta_txt}"])[0]
    melhores = sorted(candidatos, key=lambda i: _similaridade(consulta, cache["vetores"][i]),
                      reverse=True)[:4]
    # tabelas longas costumam continuar no trecho seguinte: inclui o vizinho
    # dos 2 melhores trechos para nao cortar a lista no meio
    for i in list(melhores[:2]):
        if i + 1 < len(pedacos) and i + 1 not in melhores:
            melhores.append(i + 1)
    inicio = pedacos[0][:600]
    trechos = "\n---\n".join(pedacos[i] for i in sorted(melhores))
    return (f"Artigo: {titulo} (https://pt.wikipedia.org/wiki/{urllib.parse.quote(titulo.replace(' ', '_'))})\n"
            f"Início do artigo:\n{inicio}\n\nTrechos mais relevantes para '{pergunta or termo}':\n{trechos}")

# =============================================================
# SEUS ARQUIVOS (RAG): a Jaci le os arquivos da pasta
# "documentos", divide em pedacos e guarda um "indice" com o
# significado de cada pedaco (embeddings). Na hora da pergunta,
# ele busca os pedacos mais parecidos e responde com base neles.
# =============================================================

PASTA_DOCS = "documentos"
ARQ_INDICE = "indice.json"
MODELO_EMBED = "embeddinggemma"
MODELO_VISAO = "qwen3.5:9b"   # o mesmo cerebro, usado para "ler" imagens (OCR)
TAM_PEDACO = 900       # caracteres por pedaco
SOBREPOSICAO = 150     # repete um pouco entre pedacos para nao cortar ideias

_indice = {"arquivos": {}, "pedacos": []}


def _textos_xml(elemento):
    """Junta todos os paragrafos de texto (tags a:p / a:t) de um XML do Office."""
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    linhas = []
    for paragrafo in elemento.iter(ns + "p"):
        texto = "".join(t.text or "" for t in paragrafo.iter(ns + "t"))
        if texto.strip():
            linhas.append(texto)
    return "\n".join(linhas)


def _ler_imagem(imagem, onde):
    """OCR: pede para o modelo de visao transcrever o texto de uma imagem."""
    if imagem.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        return ""
    print(f"    lendo a imagem do {onde} (OCR, pode levar 1 minuto)...", flush=True)
    resp = ollama.chat(model=MODELO_VISAO, think=False, messages=[{
        "role": "user",
        "content": "Transcreva TODO o texto desta imagem, exatamente como está "
                   "escrito, no idioma original. Mantenha listas e tabelas. "
                   "Não comente, não resuma, não traduza.",
        "images": [imagem.blob],
    }])
    return resp["message"]["content"]


def _ler_arquivo(caminho):
    """Devolve uma lista de (pagina, texto) do arquivo."""
    nome = caminho.lower()
    if nome.endswith(".pdf"):
        from pypdf import PdfReader
        leitor = PdfReader(caminho)
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(leitor.pages)]
    if nome.endswith(".pptx"):
        # cada slide vira uma "pagina". Le o XML do slide inteiro, entao pega
        # tambem textos dentro de grupos, tabelas e SmartArt, mais as notas.
        from lxml import etree
        from pptx import Presentation
        paginas = []
        for i, slide in enumerate(Presentation(caminho).slides, 1):
            partes = [_textos_xml(slide._element)]
            for rel in slide.part.rels.values():
                if "diagramData" in rel.reltype:   # SmartArt
                    partes.append(_textos_xml(etree.fromstring(rel.target_part.blob)))
            if slide.has_notes_slide:
                partes.append(slide.notes_slide.notes_text_frame.text)
            texto = "\n".join(p for p in partes if p.strip())
            # Slide sem texto? Entao provavelmente e uma imagem: usa OCR.
            if len(texto) < 20:
                for forma in slide.shapes:
                    if forma.shape_type == 13:   # 13 = PICTURE
                        texto += "\n" + _ler_imagem(forma.image, f"slide {i}")
            paginas.append((i, texto))
        return paginas
    if nome.endswith(".docx"):
        from docx import Document
        doc = Document(caminho)
        texto = "\n".join(p.text for p in doc.paragraphs)
        for tabela in doc.tables:
            for linha in tabela.rows:
                texto += "\n" + " | ".join(c.text for c in linha.cells)
        return [(1, texto)]
    with open(caminho, encoding="utf-8", errors="replace") as f:
        return [(1, f.read())]


def _dividir(texto):
    texto = " ".join(texto.split())
    passo = TAM_PEDACO - SOBREPOSICAO
    return [texto[i:i + TAM_PEDACO] for i in range(0, max(len(texto), 1), passo)
            if texto[i:i + TAM_PEDACO].strip()]


def _embeddings(textos):
    vetores = []
    for i in range(0, len(textos), 32):
        resp = ollama.embed(model=MODELO_EMBED, input=textos[i:i + 32])
        vetores += resp["embeddings"]
    return vetores


def indexar(mostrar=True):
    """Le arquivos novos ou alterados da pasta documentos e atualiza o indice."""
    global _indice
    os.makedirs(PASTA_DOCS, exist_ok=True)
    _indice = carregar(ARQ_INDICE, {"arquivos": {}, "pedacos": []})
    tipos = ("*.pdf", "*.pptx", "*.docx", "*.txt", "*.md")
    caminhos = sorted(c for t in tipos
                      for c in glob.glob(os.path.join(PASTA_DOCS, "**", t), recursive=True)
                      if not os.path.basename(c).startswith("~$"))  # ignora temporarios do Office
    atuais = {os.path.relpath(c, PASTA_DOCS): os.path.getmtime(c) for c in caminhos}

    # remove do indice arquivos apagados ou alterados
    mudou = [n for n, t in _indice["arquivos"].items() if atuais.get(n) != t]
    if mudou:
        _indice["pedacos"] = [p for p in _indice["pedacos"] if p["arquivo"] not in mudou]
        for n in mudou:
            del _indice["arquivos"][n]

    novos = [n for n in atuais if n not in _indice["arquivos"]]
    for nome in novos:
        if mostrar:
            print(f"  lendo {nome}...", flush=True)
        try:
            pedacos = []
            for pagina, texto in _ler_arquivo(os.path.join(PASTA_DOCS, nome)):
                for trecho in _dividir(texto):
                    pedacos.append({"arquivo": nome, "pagina": pagina, "texto": trecho})
            textos = [f"title: {p['arquivo']} | text: {p['texto']}" for p in pedacos]
            for p, v in zip(pedacos, _embeddings(textos)):
                p["vetor"] = v
            if not pedacos and mostrar:
                print(f"  [aviso: nao achei texto em {nome}. Se o conteudo for imagem"
                      " (slide escaneado, print), a Jaci nao consegue ler]")
            _indice["pedacos"] += pedacos
            _indice["arquivos"][nome] = atuais[nome]
        except Exception as erro:
            print(f"  [nao consegui ler {nome}: {erro}]")
    if novos or mudou:
        salvar(ARQ_INDICE, _indice)
    return len(_indice["arquivos"]), len(_indice["pedacos"])


def _similaridade(a, b):
    produto = sum(x * y for x, y in zip(a, b))
    return produto / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)


def buscar_documentos(pergunta: str) -> str:
    """Busca nos arquivos pessoais do Octavio (PDFs, anotações, materiais das aulas). Use para qualquer pergunta sobre os documentos, trabalhos ou matérias dele.

    Args:
        pergunta: O que procurar nos documentos, em poucas palavras.
    """
    if not _indice["pedacos"]:
        return "Não há documentos indexados. Peça ao Octavio para colocar arquivos na pasta documentos."
    consulta = _embeddings([f"task: search result | query: {pergunta}"])[0]
    notas = sorted(_indice["pedacos"],
                   key=lambda p: _similaridade(consulta, p["vetor"]), reverse=True)[:4]
    return "\n\n".join(f"[{p['arquivo']}, página/slide {p['pagina']}]\n{p['texto']}" for p in notas)


# =============================================================
# EXECUTAR PYTHON: a Jaci escreve um codigo e, SE o Octavio
# aprovar, roda numa pasta separada (area_python) e ve o resultado.
# =============================================================

PASTA_PYTHON = "area_python"

# Funcao que pergunta ao Octavio se pode rodar o codigo. O terminal e a
# interface trocam por uma pergunta deles. Sem ninguem para aprovar
# (por exemplo, no avaliar.py), o codigo NAO e executado.
aprovar_codigo = None


def executar_python(codigo: str) -> str:
    """Executa código Python no computador do Octavio, depois que ele aprovar. Use para análise de dados (pandas), gráficos (matplotlib: salve com plt.savefig("nome.png")), testar algoritmos e cálculos complexos. Use print() para mostrar resultados. SALVE arquivos e gráficos SEMPRE na pasta atual, só com o nome (ex: plt.savefig("grafico.png")), nunca em outra pasta. Os documentos do Octavio podem ser LIDOS em "../documentos", mas nunca escreva nem apague nada lá.

    Args:
        codigo: O código Python completo a executar.
    """
    os.makedirs(PASTA_PYTHON, exist_ok=True)
    antes = {n: os.path.getmtime(os.path.join(PASTA_PYTHON, n)) for n in os.listdir(PASTA_PYTHON)}
    # tambem vigia a pasta principal e a de documentos, para avisar se o codigo mexeu nelas
    vigiadas = {p: {n: os.path.getmtime(os.path.join(p, n)) for n in os.listdir(p)}
                for p in (".", PASTA_DOCS) if os.path.isdir(p)}
    script = os.path.join(PASTA_PYTHON, "_ultimo_codigo.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(codigo)
    ambiente = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run([sys.executable, "_ultimo_codigo.py"], cwd=PASTA_PYTHON,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120, env=ambiente)
    except subprocess.TimeoutExpired:
        return "O código demorou mais de 2 minutos e foi interrompido."
    novos = sorted(n for n in os.listdir(PASTA_PYTHON)
                   if n != "_ultimo_codigo.py" and
                   antes.get(n) != os.path.getmtime(os.path.join(PASTA_PYTHON, n)))
    partes = []
    if r.stdout.strip():
        partes.append("Saída:\n" + r.stdout[-4000:])
    if r.stderr.strip():
        partes.append("Erros:\n" + r.stderr[-2000:])
    if novos:
        partes.append("Arquivos criados/alterados em area_python: " + ", ".join(novos))
    fora = [os.path.join(p, n) for p, conteudo in vigiadas.items() for n in os.listdir(p)
            if os.path.isfile(os.path.join(p, n)) and
            conteudo.get(n) != os.path.getmtime(os.path.join(p, n))]
    if fora:
        partes.append("ATENÇÃO: o código criou ou alterou arquivos FORA da area_python: "
                      + ", ".join(fora) + ". Avise o Octavio e, nas próximas vezes, salve só na pasta atual.")
    if not partes:
        partes.append("O código rodou sem mostrar nada (use print para ver resultados).")
    return "\n\n".join(partes)


FERRAMENTAS = {f.__name__: f for f in [clima, hora_atual, calcular, buscar_wikipedia,
                                         buscar_documentos, executar_python]}


def carregar(arquivo, padrao):
    if os.path.exists(arquivo):
        with open(arquivo, encoding="utf-8") as f:
            return json.load(f)
    return padrao


def salvar(arquivo, dados):
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def montar_sistema(memoria):
    # A "memoria" entra no prompt de sistema: o modelo le isso
    # antes de toda resposta, como se fossem anotacoes sobre voce.
    texto = REGRAS
    texto += f"\n\nAgora é {hora_atual()} (horário do computador do Octavio)."
    if memoria:
        texto += "\n\nO que você sabe sobre o Octavio:\n"
        texto += "\n".join(f"- {fato}" for fato in memoria)
    return texto


GRAFIA_BR = {
    "heterónimo": "heterônimo", "pseudónimo": "pseudônimo", "anónimo": "anônimo",
    "sinónimo": "sinônimo", "antónimo": "antônimo", "fenómeno": "fenômeno",
    "económic": "econômic", "académic": "acadêmic", "género": "gênero",
    "António": "Antônio", "ténis": "tênis", "bebé": "bebê",
    "registo": "registro", "equipa ": "equipe ", "ecrã": "tela", "telemóvel": "celular",
    "pequeno-almoço": "café da manhã", "autocarro": "ônibus",
}


def _grafia_br(texto):
    for pt, br in GRAFIA_BR.items():
        texto = texto.replace(pt, br).replace(pt.capitalize(), br.capitalize())
    return texto


def responder_em_partes(mensagens):
    """Igual a _responder_bruto, mas corrige grafias de Portugal no texto.
    Segura o final de cada pedaco ate o proximo espaco, para nao quebrar
    uma palavra no meio da correcao."""
    guardado = ""
    for parte in _responder_bruto(mensagens):
        if parte["tipo"] != "texto":
            if guardado:
                yield {"tipo": "texto", "conteudo": _grafia_br(guardado)}
                guardado = ""
            yield parte
            continue
        guardado += parte["conteudo"]
        corte = max(guardado.rfind(" "), guardado.rfind("\n"))
        if corte > 0:
            yield {"tipo": "texto", "conteudo": _grafia_br(guardado[:corte])}
            guardado = guardado[corte:]
    if guardado:
        yield {"tipo": "texto", "conteudo": _grafia_br(guardado)}


def _responder_bruto(mensagens):
    """Gera a resposta aos poucos. Cada item e um dict:
    {"tipo": "texto", "conteudo": "..."} ou
    {"tipo": "ferramenta", "nome": "...", "argumentos": {...}}.
    Usado pelo terminal e pela interface (jaci_app.py)."""
    # Pode dar varias "voltas": o modelo pede uma ferramenta, o programa
    # roda, devolve o resultado, e o modelo continua ate responder.
    for _ in range(5):
        resposta = ""
        pedidos = []
        fluxo = ollama.chat(model=MODELO, messages=mensagens, stream=True,
                            think=False, tools=list(FERRAMENTAS.values()))
        for pedaco in fluxo:
            msg = pedaco.message
            if msg.content:
                resposta += msg.content
                yield {"tipo": "texto", "conteudo": msg.content}
            if msg.tool_calls:
                pedidos += msg.tool_calls
        if not pedidos:
            return
        mensagens.append({"role": "assistant", "content": resposta,
                          "tool_calls": pedidos})
        for pedido in pedidos:
            nome = pedido.function.name
            argumentos = dict(pedido.function.arguments or {})
            yield {"tipo": "ferramenta", "nome": nome, "argumentos": argumentos}
            try:
                if nome == "executar_python" and not (
                        aprovar_codigo and aprovar_codigo(argumentos.get("codigo", ""))):
                    resultado = ("O Octavio NÃO autorizou rodar este código. "
                                 "Não tente de novo; explique o que o código faria ou pergunte o que ele prefere.")
                else:
                    resultado = FERRAMENTAS[nome](**argumentos)
            except Exception as erro:
                resultado = f"Erro na ferramenta {nome}: {erro}"
            if nome == "executar_python":
                yield {"tipo": "resultado", "nome": nome, "conteudo": str(resultado)}
            mensagens.append({"role": "tool", "content": str(resultado),
                              "tool_name": nome})


def conversar(mensagens):
    resposta, usadas = "", []
    for parte in responder_em_partes(mensagens):
        if parte["tipo"] == "ferramenta":
            usadas.append(parte["nome"])
        if parte["tipo"] == "texto":
            resposta += parte["conteudo"]
            print(parte["conteudo"], end="", flush=True)
        elif parte["tipo"] == "resultado":
            print("\n  [resultado do codigo]\n  " + parte["conteudo"].replace("\n", "\n  "))
            print("Jaci: ", end="", flush=True)
        else:
            if parte["nome"] == "executar_python":
                print("\n  [a Jaci quer rodar codigo Python]", flush=True)
            else:
                args = ", ".join(f"{k}={v!r}" for k, v in parte["argumentos"].items())
                print(f"\n  [usando ferramenta: {parte['nome']}({args})]", flush=True)
                print("Jaci: ", end="", flush=True)
            resposta = ""   # guarda so o texto da resposta final
    print("\n")
    return resposta, usadas


def historico_para_mensagens(historico):
    """Prepara o historico para o modelo. As respostas que usaram ferramentas
    ganham uma anotacao, para o modelo lembrar que a acao foi FEITA com a
    ferramenta (e nao imitar respostas que "dizem" que fizeram algo)."""
    mensagens = []
    for m in historico[-MAX_HISTORICO:]:
        texto = m["content"]
        if m.get("ferramentas"):
            texto += "\n\n[nesta resposta foram usadas as ferramentas: " + ", ".join(m["ferramentas"]) + "]"
        mensagens.append({"role": m["role"], "content": texto})
    return mensagens


def ajuda():
    print("Comandos:")
    print("  /lembrar <fato>   guarda um fato na memoria")
    print("  /memoria          mostra tudo o que ele lembra")
    print("  /esquecer <n>     apaga o fato numero n")
    print("  /novo             comeca uma conversa nova (mantem a memoria)")
    print("  /indexar          rele a pasta documentos (arquivos novos ou alterados)")
    print("  /sair             fecha (a conversa fica salva)\n")


def _aprovar_no_terminal(codigo):
    print("\n" + "-" * 60 + "\n" + codigo + "\n" + "-" * 60)
    return input("Rodar este codigo? (s/n): ").strip().lower() in ("s", "sim", "y")


def main():
    global aprovar_codigo
    aprovar_codigo = _aprovar_no_terminal
    memoria = carregar(ARQ_MEMORIA, [])
    historico = carregar(ARQ_HISTORICO, [])
    print("Conferindo a pasta documentos...")
    try:
        n_arquivos, n_pedacos = indexar()
    except Exception as erro:
        print(f"[Nao consegui indexar os documentos: {erro}]")
        n_arquivos, n_pedacos = 0, 0

    print("=" * 60)
    print(" Jaci - sua assistente pessoal")
    print(f" Memoria: {len(memoria)} fatos | "
          f"Historico: {len(historico)} mensagens | "
          f"Documentos: {n_arquivos} ({n_pedacos} trechos)")
    print(" Digite /ajuda para ver os comandos")
    print("=" * 60 + "\n")

    while True:
        try:
            entrada = input("Voce: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not entrada:
            continue

        # ---------- comandos ----------
        if entrada == "/sair":
            break
        if entrada == "/ajuda":
            ajuda()
            continue
        if entrada.startswith("/lembrar"):
            fato = entrada[len("/lembrar"):].strip()
            if fato:
                memoria.append(fato)
                salvar(ARQ_MEMORIA, memoria)
                print(f"Jaci: Anotado! ({len(memoria)} fatos na memoria)\n")
            else:
                print("Use assim: /lembrar eu moro em Barcelos\n")
            continue
        if entrada == "/memoria":
            if not memoria:
                print("Jaci: Ainda nao sei nada sobre voce.\n")
            for i, fato in enumerate(memoria, 1):
                print(f"  {i}. {fato}")
            print()
            continue
        if entrada.startswith("/esquecer"):
            try:
                n = int(entrada.split()[1])
                apagado = memoria.pop(n - 1)
                salvar(ARQ_MEMORIA, memoria)
                print(f"Jaci: Esqueci: {apagado}\n")
            except (IndexError, ValueError):
                print("Use assim: /esquecer 2  (veja os numeros com /memoria)\n")
            continue
        if entrada == "/indexar":
            n_arquivos, n_pedacos = indexar()
            print(f"Jaci: Pronto! {n_arquivos} documentos, {n_pedacos} trechos.\n")
            continue
        if entrada == "/novo":
            historico = []
            salvar(ARQ_HISTORICO, historico)
            print("Jaci: Conversa nova! (a memoria continua)\n")
            continue

        # ---------- conversa normal ----------
        historico.append({"role": "user", "content": entrada})
        mensagens = [{"role": "system", "content": montar_sistema(memoria)}]
        mensagens += historico_para_mensagens(historico)

        print("Jaci: ", end="", flush=True)
        try:
            resposta, usadas = conversar(mensagens)
        except Exception as erro:
            print(f"\n[Erro ao falar com o Ollama: {erro}]")
            print("[Confira se o Ollama esta aberto e se o modelo 'jaci' existe]\n")
            historico.pop()
            continue

        historico.append({"role": "assistant", "content": resposta, "ferramentas": usadas})
        salvar(ARQ_HISTORICO, historico[-MAX_HISTORICO:])

    print("Jaci: Ate mais! Nossa conversa ficou salva.")


if __name__ == "__main__":
    main()
