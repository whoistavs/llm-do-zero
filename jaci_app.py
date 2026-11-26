import json
import mimetypes
import os
import threading
import uuid
import urllib.parse
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jaci as t

# =============================================================
# JACI APP - interface de chat no navegador
# -------------------------------------------------------------
# Roda um pequeno servidor so no seu computador (localhost) e abre
# a pagina de chat. Usa a mesma memoria, historico, ferramentas e
# documentos do jaci.py.
# =============================================================

PORTA = 8765
ARQ_ERROS = "erros_reportados.json"   # respostas marcadas com 👎
trava = threading.Lock()   # uma resposta de cada vez
pendentes = {}             # codigos esperando o Octavio aprovar

estado = {
    "memoria": t.carregar(t.ARQ_MEMORIA, []),
    "historico": t.carregar(t.ARQ_HISTORICO, []),
}


def info_docs():
    return {"arquivos": len(t._indice["arquivos"]), "trechos": len(t._indice["pedacos"])}


class Servidor(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, dados, codigo=200):
        corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def _ler_corpo(self):
        tamanho = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(tamanho) or b"{}")

    def do_GET(self):
        if self.path == "/":
            corpo = PAGINA.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)
        elif self.path.startswith("/area_python/"):
            nome = os.path.basename(urllib.parse.unquote(self.path.split("?")[0]))
            caminho = os.path.join(t.PASTA_PYTHON, nome)
            if not os.path.isfile(caminho):
                return self._json({"erro": "nao encontrado"}, 404)
            with open(caminho, "rb") as f:
                corpo = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(nome)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)
        elif self.path == "/api/estado":
            self._json({"memoria": estado["memoria"],
                        "historico": estado["historico"],
                        "docs": info_docs()})
        else:
            self._json({"erro": "nao encontrado"}, 404)

    def do_POST(self):
        dados = self._ler_corpo()
        if self.path == "/api/chat":
            self._chat(dados.get("mensagem", "").strip())
        elif self.path == "/api/lembrar":
            fato = dados.get("fato", "").strip()
            if fato:
                estado["memoria"].append(fato)
                t.salvar(t.ARQ_MEMORIA, estado["memoria"])
            self._json({"memoria": estado["memoria"]})
        elif self.path == "/api/esquecer":
            i = dados.get("indice", -1)
            if 0 <= i < len(estado["memoria"]):
                estado["memoria"].pop(i)
                t.salvar(t.ARQ_MEMORIA, estado["memoria"])
            self._json({"memoria": estado["memoria"]})
        elif self.path == "/api/aprovar":
            pedido = pendentes.get(dados.get("id"))
            if pedido:
                pedido["ok"] = bool(dados.get("ok"))
                pedido["evento"].set()
            self._json({"ok": True})
        elif self.path == "/api/reportar":
            # guarda o erro para virar teste no avaliar.py depois
            erros = t.carregar(ARQ_ERROS, [])
            erros.append({"data": datetime.now().strftime("%d/%m/%Y %H:%M"),
                          "pergunta": dados.get("pergunta", ""),
                          "resposta": dados.get("resposta", ""),
                          "comentario": dados.get("comentario", "")})
            t.salvar(ARQ_ERROS, erros)
            self._json({"total": len(erros)})
        elif self.path == "/api/novo":
            estado["historico"] = []
            t.salvar(t.ARQ_HISTORICO, [])
            self._json({"ok": True})
        elif self.path == "/api/indexar":
            with trava:
                t.indexar()
            self._json({"docs": info_docs()})
        else:
            self._json({"erro": "nao encontrado"}, 404)

    def _chat(self, mensagem):
        # Responde em "streaming": uma linha JSON por pedaco de texto
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def enviar(obj):
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        if not mensagem:
            enviar({"tipo": "fim"})
            return
        def aprovar(codigo):
            # mostra o codigo na pagina e espera o Octavio clicar (ate 10 min)
            pid = uuid.uuid4().hex
            pendentes[pid] = {"evento": threading.Event(), "ok": False}
            enviar({"tipo": "aprovacao", "id": pid, "codigo": codigo})
            pendentes[pid]["evento"].wait(timeout=600)
            return pendentes.pop(pid)["ok"]

        with trava:
            t.aprovar_codigo = aprovar
            historico = estado["historico"]
            historico.append({"role": "user", "content": mensagem})
            mensagens = [{"role": "system", "content": t.montar_sistema(estado["memoria"])}]
            mensagens += t.historico_para_mensagens(historico)
            resposta, usadas = "", []
            try:
                for parte in t.responder_em_partes(mensagens):
                    if parte["tipo"] == "texto":
                        resposta += parte["conteudo"]
                    elif parte["tipo"] == "ferramenta":
                        resposta = ""   # guarda so o texto final
                        usadas.append(parte["nome"])
                    enviar(parte)
                historico.append({"role": "assistant", "content": resposta, "ferramentas": usadas})
                estado["historico"] = historico[-t.MAX_HISTORICO:]
                t.salvar(t.ARQ_HISTORICO, estado["historico"])
            except (BrokenPipeError, ConnectionResetError):
                historico.pop()
                return
            except Exception as erro:
                historico.pop()
                enviar({"tipo": "erro", "conteudo": str(erro)})
        enviar({"tipo": "fim"})


PAGINA = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jaci</title>
<style>
  :root {
    --fundo: #0f1115; --painel: #161920; --borda: #262a33; --texto: #e7e9ee;
    --suave: #8b919e; --destaque: #6aa8ff; --voce: #1f3b63; --erro: #ff7a7a;
  }
  * { box-sizing: border-box; }
  body { margin: 0; height: 100vh; display: flex; background: var(--fundo);
         color: var(--texto); font: 15px/1.55 "Segoe UI", system-ui, sans-serif; }
  aside { width: 290px; background: var(--painel); border-right: 1px solid var(--borda);
          padding: 20px 16px; display: flex; flex-direction: column; gap: 18px; overflow-y: auto; }
  aside h1 { font-size: 20px; margin: 0; }
  aside h1 span { color: var(--suave); font-weight: 400; font-size: 13px; display: block; }
  aside h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
             color: var(--suave); margin: 0 0 8px; }
  button { background: #222733; color: var(--texto); border: 1px solid var(--borda);
           border-radius: 8px; padding: 8px 12px; cursor: pointer; font: inherit; font-size: 13px; }
  button:hover { border-color: var(--destaque); }
  button:disabled { opacity: .5; cursor: default; }
  .memoria li { display: flex; gap: 6px; align-items: flex-start; font-size: 13px;
                padding: 6px 0; border-bottom: 1px solid var(--borda); }
  .memoria li span { flex: 1; }
  .memoria li button { padding: 0 6px; border: none; background: none; color: var(--suave); }
  .memoria li button:hover { color: var(--erro); }
  .memoria ul { list-style: none; margin: 0; padding: 0; }
  .linha { display: flex; gap: 6px; margin-top: 8px; }
  .linha input { flex: 1; min-width: 0; }
  input, textarea { background: var(--fundo); color: var(--texto); border: 1px solid var(--borda);
                    border-radius: 8px; padding: 8px 10px; font: inherit; }
  input:focus, textarea:focus { outline: none; border-color: var(--destaque); }
  .docs { font-size: 13px; color: var(--suave); margin-bottom: 8px; }
  main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  #conversa { flex: 1; overflow-y: auto; padding: 28px max(24px, calc(50% - 380px)); }
  .msg { margin: 0 0 18px; display: flex; }
  .msg .bolha { padding: 10px 14px; border-radius: 14px; max-width: 100%; overflow-wrap: anywhere; }
  .voce { justify-content: flex-end; }
  .voce .bolha { background: var(--voce); max-width: 80%; white-space: pre-wrap; }
  .ia .bolha { background: transparent; padding-left: 0; }
  .ia .bolha p { margin: 0 0 10px; }
  .ia .bolha p:last-child { margin-bottom: 0; }
  .ia .bolha ul, .ia .bolha ol { margin: 0 0 10px; padding-left: 22px; }
  .ia .bolha code { background: #222733; padding: 1px 5px; border-radius: 5px; font-size: 13px; }
  .ia .bolha pre { background: #0a0c10; border: 1px solid var(--borda); padding: 12px;
                   border-radius: 8px; overflow-x: auto; }
  .ia .bolha pre code { background: none; padding: 0; }
  .ferramenta { font-size: 12px; color: var(--destaque); background: #15233a;
                border-radius: 999px; padding: 3px 10px; display: inline-block; margin: 0 0 8px; }
  .erro { color: var(--erro); }
  .aprovacao { border: 1px solid #5a4a1a; background: #1d1a10; border-radius: 10px;
               padding: 10px 12px; margin: 4px 0 10px; }
  .aprovacao .titulo { font-size: 13px; color: #f0c85a; margin-bottom: 6px; }
  .aprovacao .botoes { display: flex; gap: 8px; margin-top: 8px; }
  .aprovacao .rodar { background: #2f6b3a; border-color: #2f6b3a; }
  .resultado { border: 1px solid var(--borda); border-radius: 10px; padding: 8px 12px; margin: 0 0 10px; }
  .resultado summary { cursor: pointer; font-size: 13px; color: var(--suave); }
  .resultado pre { white-space: pre-wrap; font-size: 12px; max-height: 300px; overflow-y: auto; }
  .resultado img { max-width: 100%; border-radius: 8px; margin-top: 8px; background: #fff; }
  .acoes { margin-top: 6px; }
  .acoes button { padding: 2px 8px; font-size: 12px; background: none; color: var(--suave);
                  border-color: transparent; }
  .acoes button:hover { border-color: var(--erro); color: var(--erro); }
  .acoes .ok { color: var(--suave); font-size: 12px; }
  .digitando::after { content: "▍"; animation: pisca 1s steps(2) infinite; color: var(--suave); }
  @keyframes pisca { 50% { opacity: 0; } }
  .vazio { color: var(--suave); text-align: center; margin-top: 18vh; }
  .vazio b { display: block; font-size: 26px; color: var(--texto); margin-bottom: 6px; }
  form#envio { display: flex; gap: 10px; padding: 16px max(24px, calc(50% - 380px)) 24px; }
  form#envio textarea { flex: 1; resize: none; height: 52px; max-height: 200px; }
  form#envio button { background: var(--destaque); color: #08111f; border: none;
                      font-weight: 600; padding: 0 20px; }
</style>
</head>
<body>
<aside>
  <h1>Jaci<span>assistente pessoal · 100% local</span></h1>
  <div><button id="novo">＋ Nova conversa</button></div>
  <div class="memoria">
    <h2>Memória</h2>
    <ul id="lista-memoria"></ul>
    <div class="linha">
      <input id="novo-fato" placeholder="Algo pra eu lembrar…">
      <button id="lembrar">Salvar</button>
    </div>
  </div>
  <div>
    <h2>Documentos</h2>
    <div class="docs" id="docs">…</div>
    <button id="indexar">Reler pasta documentos</button>
  </div>
</aside>
<main>
  <div id="conversa"></div>
  <form id="envio">
    <textarea id="texto" placeholder="Pergunte qualquer coisa… (Enter envia, Shift+Enter pula linha)"></textarea>
    <button id="enviar" type="submit">Enviar</button>
  </form>
</main>
<script>
const $ = (id) => document.getElementById(id);
const conversa = $("conversa");
let ocupado = false;

function escapar(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Markdown simples: titulos, negrito, italico, codigo, listas e paragrafos
function markdown(texto) {
  const blocos = [];
  texto = texto.replace(/```[\w-]*\n?([\s\S]*?)```/g, (_, c) => {
    blocos.push("<pre><code>" + escapar(c) + "</code></pre>");
    return "\u0000" + (blocos.length - 1) + "\u0000";
  });
  const inline = (s) => escapar(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>");
  let html = "", lista = null;
  const fechar = () => { if (lista) { html += "</" + lista + ">"; lista = null; } };
  for (const linha of texto.split("\n")) {
    const num = linha.match(/^\s*\d+[.)]\s+(.*)/);
    const pt = linha.match(/^\s*[-*•]\s+(.*)/);
    const tit = linha.match(/^#{1,6}\s+(.*)/);
    const bloco = linha.match(/^\u0000(\d+)\u0000$/);
    if (num || pt) {
      const tipo = num ? "ol" : "ul";
      if (lista !== tipo) { fechar(); html += "<" + tipo + ">"; lista = tipo; }
      html += "<li>" + inline((num || pt)[1]) + "</li>";
    } else {
      fechar();
      if (bloco) html += blocos[+bloco[1]];
      else if (tit) html += "<p><b>" + inline(tit[1]) + "</b></p>";
      else if (linha.trim()) html += "<p>" + inline(linha) + "</p>";
    }
  }
  fechar();
  return html;
}

function adicionar(quem, conteudo) {
  const vazio = conversa.querySelector(".vazio");
  if (vazio) vazio.remove();
  const div = document.createElement("div");
  div.className = "msg " + quem;
  const bolha = document.createElement("div");
  bolha.className = "bolha";
  if (quem === "voce") bolha.textContent = conteudo;
  else bolha.innerHTML = markdown(conteudo);
  div.appendChild(bolha);
  conversa.appendChild(div);
  conversa.scrollTop = conversa.scrollHeight;
  return bolha;
}

// Botao 👎: guarda a pergunta, a resposta e o que estava errado
function botaoFeedback(bolha, pergunta, resposta) {
  const acoes = document.createElement("div");
  acoes.className = "acoes";
  const b = document.createElement("button");
  b.textContent = "👎 Resposta errada";
  b.onclick = async () => {
    const comentario = prompt("O que estava errado? Qual seria a resposta certa?");
    if (comentario === null) return;
    const r = await post("/api/reportar", { pergunta, resposta, comentario });
    acoes.innerHTML = '<span class="ok">Obrigado! Erro salvo para virar teste (' + r.total + ' no total).</span>';
  };
  acoes.appendChild(b);
  bolha.parentElement.appendChild(acoes);
  bolha.parentElement.style.flexDirection = "column";
}

function telaVazia() {
  conversa.innerHTML = '<div class="vazio"><b>Oi, Octavio!</b>Em que posso ajudar hoje?</div>';
}

function mostrarMemoria(memoria) {
  const ul = $("lista-memoria");
  ul.innerHTML = "";
  if (!memoria.length) ul.innerHTML = '<li><span style="color:var(--suave)">Nada ainda.</span></li>';
  memoria.forEach((fato, i) => {
    const li = document.createElement("li");
    li.innerHTML = "<span></span><button title='Esquecer'>✕</button>";
    li.querySelector("span").textContent = fato;
    li.querySelector("button").onclick = async () => {
      const r = await post("/api/esquecer", { indice: i });
      mostrarMemoria(r.memoria);
    };
    ul.appendChild(li);
  });
}

function mostrarDocs(d) {
  $("docs").textContent = d.arquivos + " arquivo(s), " + d.trechos + " trecho(s)";
}

async function post(url, dados) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                               body: JSON.stringify(dados || {}) });
  return r.json();
}

async function carregar() {
  const e = await (await fetch("/api/estado")).json();
  mostrarMemoria(e.memoria);
  mostrarDocs(e.docs);
  conversa.innerHTML = "";
  if (!e.historico.length) telaVazia();
  let ultimaPergunta = "";
  e.historico.forEach((m) => {
    if (m.role === "user") { ultimaPergunta = m.content; adicionar("voce", m.content); }
    else botaoFeedback(adicionar("ia", m.content), ultimaPergunta, m.content);
  });
}

async function enviar(texto) {
  if (ocupado || !texto.trim()) return;
  ocupado = true;
  $("enviar").disabled = true;
  adicionar("voce", texto);
  const bolha = adicionar("ia", "");
  bolha.classList.add("digitando");
  // "extras" guarda ferramentas, aprovacoes e resultados (nao sao redesenhados)
  const extras = document.createElement("div");
  const textoDiv = document.createElement("div");
  bolha.appendChild(extras);
  bolha.appendChild(textoDiv);
  let resposta = "";
  const rolar = () => { conversa.scrollTop = conversa.scrollHeight; };
  const desenhar = () => { textoDiv.innerHTML = markdown(resposta); rolar(); };
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify({ mensagem: texto }) });
    const leitor = r.body.getReader();
    const dec = new TextDecoder();
    let sobra = "";
    while (true) {
      const { value, done } = await leitor.read();
      if (done) break;
      sobra += dec.decode(value, { stream: true });
      const linhas = sobra.split("\n");
      sobra = linhas.pop();
      for (const linha of linhas) {
        if (!linha.trim()) continue;
        const p = JSON.parse(linha);
        if (p.tipo === "texto") { resposta += p.conteudo; desenhar(); }
        else if (p.tipo === "ferramenta") {
          const chip = document.createElement("div");
          chip.className = "ferramenta";
          if (p.nome === "executar_python") chip.textContent = "🐍 código Python";
          else {
            const args = Object.values(p.argumentos).join(", ");
            chip.textContent = "🔧 " + p.nome + (args ? ": " + args : "");
          }
          extras.appendChild(chip);
          extras.appendChild(document.createElement("br"));
          resposta = ""; desenhar();
        } else if (p.tipo === "aprovacao") {
          extras.appendChild(caixaAprovacao(p));
          rolar();
        } else if (p.tipo === "resultado") {
          extras.appendChild(caixaResultado(p.conteudo));
          rolar();
        } else if (p.tipo === "erro") {
          resposta += '\n\nErro: ' + p.conteudo + "\nConfira se o Ollama está aberto.";
          desenhar();
        }
      }
    }
  } catch (err) {
    resposta += "\n\nErro de conexão: " + err;
    desenhar();
  }
  bolha.classList.remove("digitando");
  if (resposta.trim()) botaoFeedback(bolha, texto, resposta);
  ocupado = false;
  $("enviar").disabled = false;
  $("texto").focus();
}

// Mostra o codigo que a Jaci quer rodar, com os botoes Rodar / Cancelar
function caixaAprovacao(p) {
  const caixa = document.createElement("div");
  caixa.className = "aprovacao";
  caixa.innerHTML = "<div class='titulo'>A Jaci quer rodar este código no seu computador:</div>" +
                    "<pre><code></code></pre><div class='botoes'></div>";
  caixa.querySelector("code").textContent = p.codigo;
  const botoes = caixa.querySelector(".botoes");
  const responder = async (ok) => {
    botoes.innerHTML = ok ? "<span class='ok'>✅ Aprovado, rodando…</span>"
                          : "<span class='ok'>❌ Cancelado</span>";
    await post("/api/aprovar", { id: p.id, ok });
  };
  const rodar = document.createElement("button");
  rodar.className = "rodar"; rodar.textContent = "▶ Rodar";
  rodar.onclick = () => responder(true);
  const cancelar = document.createElement("button");
  cancelar.textContent = "Cancelar";
  cancelar.onclick = () => responder(false);
  botoes.append(rodar, cancelar);
  return caixa;
}

// Mostra a saida do codigo e as imagens (graficos) que ele criou
function caixaResultado(conteudo) {
  const caixa = document.createElement("details");
  caixa.className = "resultado";
  caixa.open = true;
  caixa.innerHTML = "<summary>Resultado do código</summary><pre></pre>";
  caixa.querySelector("pre").textContent = conteudo;
  const m = conteudo.match(/Arquivos criados\/alterados em area_python: (.*)/);
  if (m) {
    for (const nome of m[1].split(", ")) {
      if (/\.(png|jpe?g|gif|svg)$/i.test(nome)) {
        const img = document.createElement("img");
        img.src = "/area_python/" + encodeURIComponent(nome) + "?t=" + Date.now();
        caixa.appendChild(img);
      }
    }
  }
  return caixa;
}

$("envio").onsubmit = (e) => { e.preventDefault(); const t = $("texto").value; $("texto").value = ""; enviar(t); };
$("texto").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("envio").requestSubmit(); }
};
$("novo").onclick = async () => { if (ocupado) return; await post("/api/novo"); telaVazia(); };
$("lembrar").onclick = async () => {
  const fato = $("novo-fato").value.trim();
  if (!fato) return;
  const r = await post("/api/lembrar", { fato });
  $("novo-fato").value = "";
  mostrarMemoria(r.memoria);
};
$("novo-fato").onkeydown = (e) => { if (e.key === "Enter") $("lembrar").click(); };
$("indexar").onclick = async () => {
  $("indexar").disabled = true;
  $("docs").textContent = "Lendo documentos… (pode demorar)";
  const r = await post("/api/indexar");
  mostrarDocs(r.docs);
  $("indexar").disabled = false;
};
carregar();
$("texto").focus();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Conferindo a pasta documentos...")
    try:
        t.indexar()
    except Exception as erro:
        print(f"[Nao consegui indexar os documentos: {erro}]")
    endereco = f"http://localhost:{PORTA}"
    print(f"\nJaci rodando em {endereco}")
    print("Deixe esta janela aberta. Para fechar, aperte Ctrl + C.\n")
    threading.Timer(1, lambda: webbrowser.open(endereco)).start()
    try:
        ThreadingHTTPServer(("127.0.0.1", PORTA), Servidor).serve_forever()
    except KeyboardInterrupt:
        print("Ate mais!")
