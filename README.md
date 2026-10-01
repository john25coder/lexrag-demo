# LexRAG Demo

API pequena para carregar documentos, dividir o texto em trechos, gerar embeddings, recuperar os
trechos mais relevantes e responder com referências verificáveis. O projeto foi pensado como uma
demonstração de tecnologia aplicada a acervos documentais e não oferece aconselhamento jurídico.

Os arquivos em `data/demo` são inteiramente fictícios. Nenhum documento confidencial ou dado de
cliente é incluído no repositório.

## O que o projeto demonstra

- Python com organização por camadas e anotações de tipo;
- API REST em FastAPI, com documentação Swagger automática;
- ingestão de PDF, TXT e Markdown;
- OCR opcional para imagens;
- embeddings locais, busca vetorial por similaridade e RAG;
- respostas com referência, trecho, arquivo, página e pontuação de relevância;
- persistência em SQLite, detecção de arquivos duplicados e testes automatizados;
- integrações opcionais com Sentence Transformers e OpenAI.

## Decisão de escopo

O modo padrão é gratuito e funciona sem chave de API. Ele usa um embedding local por hashing e uma
resposta extrativa. Esse embedding é lexical: termos semelhantes que usam palavras muito diferentes
podem não ser aproximados. O modo opcional com Sentence Transformers fornece busca semântica, e o
modo opcional com LLM redige uma resposta a partir dos trechos recuperados.

Um agente autônomo não foi incluído porque não melhora o fluxo principal deste protótipo. O pipeline
é deliberadamente previsível: ingestão, recuperação e resposta. Em um produto real, um agente faria
sentido apenas para ações controladas, como solicitar um documento ausente ou encaminhar uma tarefa
para revisão humana.

## Arquitetura

```text
arquivo -> extração de texto -> divisão em trechos -> embeddings -> SQLite
                                                               |
pergunta -> embedding da pergunta -> similaridade -> top trechos -> resposta + fontes
```

O armazenamento fica em `data/lexrag.sqlite3`, ignorado pelo Git. Cada trecho guarda o identificador
do documento, a posição, a página quando disponível, o vetor e o nome do backend que o gerou. Ao
trocar o backend de embeddings, indexe novamente os documentos para evitar vetores incompatíveis.

## Instalação rápida

Requer Python 3.11 ou superior.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.cli index data\demo
python -m app.cli ask "Qual é o prazo de retenção dos contratos encerrados?"
uvicorn app.main:app --reload
```

### Linux ou macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.cli index data/demo
python -m app.cli ask "Qual é o prazo de retenção dos contratos encerrados?"
uvicorn app.main:app --reload
```

Abra `http://127.0.0.1:8000/docs` para testar a API no navegador.

## API REST

### Indexar documentos

```bash
curl -X POST http://127.0.0.1:8000/documents/index \
  -F "files=@data/demo/politica_retencao_ficticia.md" \
  -F "files=@data/demo/procedimento_atendimento_ficticio.txt"
```

### Fazer uma pergunta

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Qual é o prazo de retenção dos contratos encerrados?","top_k":4}'
```

Resposta resumida:

```json
{
  "answer": "Com base nos documentos indexados: ... [F1]",
  "sources": [
    {
      "reference": "F1",
      "filename": "politica_retencao_ficticia.md",
      "page": null,
      "score": 0.42,
      "excerpt": "Os contratos encerrados devem permanecer..."
    }
  ]
}
```

Outras rotas:

- `GET /health`: estado e backends ativos;
- `GET /documents`: documentos e quantidade de trechos;
- `DELETE /documents/{id}`: remove um documento e seus trechos;
- `POST /query`: pergunta e fontes recuperadas.

## Busca semântica opcional

```powershell
python -m pip install -r requirements-semantic.txt
$env:RAG_EMBEDDING_BACKEND="sentence-transformers"
python -m app.cli index data\demo
```

O modelo padrão é multilíngue e é baixado na primeira execução. Para resposta redigida por LLM:

```powershell
$env:RAG_ANSWER_BACKEND="openai"
$env:OPENAI_API_KEY="sua-chave"
uvicorn app.main:app --reload
```

Use somente dados adequados ao provedor escolhido. Em contexto jurídico real, a política de
privacidade, a base legal, o controle de acesso e os contratos com fornecedores precisam ser
avaliados antes de enviar qualquer conteúdo a uma API externa.

## OCR opcional

Para imagens PNG, JPG ou TIFF, instale `requirements-ocr.txt` e o executável Tesseract com o pacote
de idioma português. PDFs com texto são processados diretamente. PDFs apenas digitalizados devem ser
convertidos em imagens neste protótipo antes do envio.

## Testes

```powershell
pytest
```

Os testes cobrem saúde da API, indexação, consulta com fonte, duplicidade e ausência de contexto. O
workflow em `.github/workflows/tests.yml` executa a suíte em cada push e pull request.

## Segurança e limites

- uploads não são gravados; apenas texto, metadados e vetores ficam no SQLite;
- documentos são tratados como dados não confiáveis no prompt opcional do LLM;
- não há autenticação, autorização, criptografia em repouso ou isolamento por cliente;
- a busca padrão favorece correspondência lexical e não substitui validação profissional;
- antes de uso real, seriam necessários autenticação, trilha de auditoria, exclusão conforme política
  de retenção, testes de prompt injection e revisão humana obrigatória.

## Demonstração em três minutos

1. Explique o fluxo do diagrama de arquitetura.
2. Rode `python -m app.cli index data/demo` e mostre os trechos criados.
3. Faça uma pergunta pelo Swagger e abra a fonte retornada.
4. Mude uma palavra no documento fictício, reindexe e mostre a detecção por hash.
5. Termine explicando a diferença entre o backend local e a busca semântica opcional.

## Ideias para evolução

- isolamento de dados por organização e perfil de acesso;
- banco vetorial com filtros por tipo, cliente, assunto e data;
- OCR de PDFs digitalizados com fila de processamento;
- avaliação automática de recuperação e fidelidade das respostas;
- interface web simples e revisão humana antes de qualquer ação externa.

