# LLM do zero: do mini-GPT ao Tavinho

Projeto pessoal para entender, na prática, como funciona uma LLM (como o ChatGPT), em duas partes:

1. **Mini-GPT do zero.** Construí um modelo de linguagem passo a passo em Python e PyTorch: de um modelo bigrama até um Transformer com atenção, treinado com os romances de Machado de Assis e depois ajustado para conversar.
2. **Tavinho.** Um assistente pessoal que roda 100% local, montado em cima de um modelo aberto (Qwen 3.5 9B via Ollama), com personalidade, memória, ferramentas, busca nos meus arquivos (RAG) e leitura de imagens (OCR).

Tudo roda num notebook comum (16 GB de RAM, sem GPU dedicada).

---

## Parte 1: Mini-GPT do zero

Cada passo é um arquivo independente e adiciona uma ideia nova.

| Passo | Arquivo | O que aprende | Resultado |
|---|---|---|---|
| 1 | `passo1_bigrama.py` | Prever a próxima letra olhando só a anterior | Erro cai de 3,85 para 2,13; inventa nomes como "Juline" e "Lela" |
| 2 | `passo2_mlp.py` | Rede neural com embeddings, camada escondida e treino em lotes | Erro 0,93; mas começa a **decorar** nomes (overfitting) |
| 3 | `passo3_livro.py` | Dados reais (3,1 milhões de caracteres de Machado de Assis) e separação treino/validação | Erro de validação 1,81; texto com cara de português, sem sentido |
| 4 | `passo4_gpt.py` | **Transformer com atenção** (4 blocos, 4 cabeças, 833 mil parâmetros) | Erro de validação 1,54; escreve diálogos com travessão, como nos romances |
| 5 | `passo5_usar.py` | Carregar o modelo salvo e gerar texto com temperatura | Continua qualquer frase "no estilo Machado" |
| 6 | `passo6_conversa.py` | **Fine-tuning de conversa**: transformar o modelo base num assistente | Responde no formato pergunta/resposta |
| 7 | `passo7_dicionario.py` | Fine-tuning com o Wikcionário (licença CC BY-SA, via kaikki.org) | Aprende regras de formação de palavras, não significados |

### O que eu aprendi

- **Fine-tuning ensina comportamento, não conhecimento.** No passo 6, o modelo aprendeu a responder como assistente, mas errou tudo o que não estava no treino: disse que a capital da Grécia é Roma e que 2 + 2 = 3.
- **Alucinação tem explicação.** Quando não sabe, o modelo gera a resposta que *parece* certa, no formato certo. No passo 7, ele "definiu" palavras que nunca viu com o padrão correto ("Gerúndio do verbo…", "Relativo ou pertencente a…") e o conteúdo inventado.
- **Mais dados não bastam sem mais parâmetros.** Com 833 mil parâmetros, o modelo guarda as regras do português (sufixos como -ndo, -ense, -mento), mas não consegue guardar milhares de definições.
- A diferença entre este mini-GPT e um ChatGPT não é a arquitetura (é a mesma): é a escala dos dados, dos parâmetros e do custo do pré-treino.

### Como rodar

```
pip install torch nltk
python passo1_bigrama.py
```

O passo 4 treina em cerca de 5 minutos na CPU e salva `mini_gpt.pt`, que os passos 5 a 7 usam.

---

## Parte 2: Tavinho, assistente pessoal local

Arquivos: `Modelfile` (personalidade) e `tavinho.py` (programa).

### Funcionalidades

1. **Modelo local** de 9 bilhões de parâmetros (Qwen 3.5 9B), rodando pelo Ollama.
2. **Personalidade e regras** no `Modelfile`: português do Brasil, respostas diretas, e admitir quando não tem certeza.
3. **Memória permanente:** fatos salvos com `/lembrar` entram no prompt de sistema de toda conversa.
4. **Ferramentas (tool calling):** o modelo decide sozinho quando usar cada uma.
   - `clima`: previsão real (Open-Meteo).
   - `hora_atual`: data e hora.
   - `calcular`: contas exatas, com avaliação segura de expressões.
   - `buscar_wikipedia`: conferir fatos antes de responder.
5. **RAG nos meus arquivos:** lê PDF, PowerPoint, Word, `.txt` e `.md` da pasta `documentos`, gera embeddings com o `embeddinggemma` e busca os trechos mais relevantes, citando arquivo e página.
6. **OCR com o próprio modelo:** slides que são só imagem são transcritos pela visão do Qwen antes de entrar no índice.

### O que eu aprendi

- **Regras no prompt mudam o comportamento, mas não dão conhecimento.** Mesmo com a regra "não invente", o modelo inventou títulos de livros até ganhar uma ferramenta para consultar.
- **Uma ferramenta só ajuda se trouxer a informação certa.** Com um resumo incompleto da Wikipédia, o modelo completou o resto inventando. Foi preciso uma regra explícita: "responda só com o que a ferramenta retornou".
- **O histórico pesa mais que as instruções.** Uma resposta antiga com LaTeX no histórico fez o modelo continuar usando LaTeX, mesmo depois de proibido. Uma conversa nova resolveu.

### Como rodar

```
# 1. Instalar o Ollama (ollama.com) e baixar os modelos
ollama pull qwen3.5:9b
ollama pull embeddinggemma

# 2. Criar o modelo com a personalidade
ollama create tavinho -f Modelfile

# 3. Instalar as dependências e rodar
pip install -r requirements.txt
python tavinho.py
```

Comandos dentro do Tavinho: `/lembrar`, `/memoria`, `/esquecer`, `/novo`, `/indexar`, `/ajuda`, `/sair`.

Memória, histórico, índice e documentos ficam só no computador (estão no `.gitignore`).

---

## Tecnologias

Python · PyTorch · Transformers (implementação própria) · Ollama · Qwen 3.5 · EmbeddingGemma · RAG · Tool calling · NLTK · pypdf · python-pptx

## Créditos

- Os passos 1 a 4 seguem a linha didática das aulas "Neural Networks: Zero to Hero", de Andrej Karpathy.
- Textos de Machado de Assis (domínio público), via corpus `machado` do NLTK.
- Definições do Wikcionário em português (CC BY-SA), extraídas pelo projeto Kaikki/wiktextract.

Feito por [Octavio Schwab](https://github.com/whoistavs).
