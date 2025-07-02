# Chat com seus Documentos (RAG)

Este projeto é uma análise e estudo de chatbot do módulo 07 do curso [IA MASTER (pycodebr)](https://pycodebr.com.br/ia-master/). A aplicação é construída com **Streamlit**, que permite ao usuário fazer upload de arquivos PDF e interagir com o conteúdo via **ChatGPT** (usando modelos da OpenAI), com base em **RAG - Retrieval-Augmented Generation**.

---
Para levar esse projeto a um canal de WhatsApp, seria necessário substituir a interface **Streamlit** por um backend em **FastAPI (ou outro framework web)** e implementar um **webhook** que receba as mensagens enviadas pelo usuário.

Nesse novo fluxo, o usuário enviaria uma mensagem ou um arquivo PDF via WhatsApp, o **webhook** receberia e processaria o conteúdo, utilizaria a mesma lógica de LangChain para recuperação e geração da resposta, e então enviaria a resposta automaticamente para o número do usuário via API. Essa abordagem manteria toda a  lógica do projeto, mas ampliaria a acessibilidade, dispensando o uso de interface web.

---

## Como funciona?

1. Você faz upload de um ou mais arquivos PDF.
2. O conteúdo é dividido em partes (chunks) e vetorizado com **OpenAI Embeddings**.
3. Os vetores são armazenados localmente usando **ChromaDB**.
4. Ao fazer uma pergunta, o sistema busca os trechos mais relevantes do conteúdo e envia para a LLM responder com base nesse contexto.

---

## Tecnologias utilizadas

- [**Streamlit**](https://streamlit.io/): cria a interface web para upload de PDFs e interação com o chatbot.
- [**LangChain**](https://www.langchain.com/): organiza o fluxo RAG com recuperação de contexto e geração de respostas.
- [**Chroma**](https://docs.trychroma.com/): banco vetorial local para armazenar e consultar os embeddings dos documentos.
- [**OpenAI API**](https://platform.openai.com/): fornece os modelos de linguagem usados para gerar as respostas.
- [**python-decouple**](https://pypi.org/project/python-decouple/): gerencia variáveis de ambiente como a chave da OpenAI.
- Python 3.9+

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/chat-documentos-rag.git
cd chat-documentos-rag
```

### 2. Crie e ative o ambiente virtual

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (CMD ou PowerShell):

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Crie o aquivo .env

Crie um arquivo .env na raiz do projeto com o seguinte conteúdo:

```ini
OPENAI_API_KEY=sua_chave_de_api...
```

Substitua sua_chave_de_api... pela sua chave de API da OpenAI.

## Executando o projeto

Depois de ativar o ambiente virtual e configurar o .env, execute:

```bash
streamlit run app.py
```

## Estrutura de diretórios

```bash
rag_chatbot/
├── app.py
├── db/                  # Persistência local dos vetores (Chroma)
├── .env                 # Sua chave de API da OpenAI
├── requirements.txt
└── README.md
```

## Exemplos de uso

- Faça upload de arquivos como contrato.pdf, artigo.pdf, manual.pdf.

- Pergunte coisas como:

- "Qual o objetivo do contrato?"

- "O artigo apresenta resultados sobre quais temas?"
