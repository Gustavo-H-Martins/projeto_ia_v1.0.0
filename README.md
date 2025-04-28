# GenAI 1.0.0

## Descrição

GenAI 1.0.0 é um agente de Inteligência Artificial projetado para analisar documentos e gerar respostas inteligentes baseadas em seu conteúdo. As principais características incluem:

- **Análise de Documentos:** Processa diversos formatos de arquivo (PDF, CSV, XLSX, XLS, TXT e JSON) para extrair informações relevantes.
- **Tecnologia RAG:** Utiliza a biblioteca LangChain com o paradigma RAG (Retrieval-Augmented Generation) para combinar técnicas de recuperação de informação e geração de texto.
- **LLMs Integrados:** Integra modelos de linguagem da Hugging Face, OpenAI e Ollama para gerar respostas contextuais. Por exemplo, pode usar GPT-4 da OpenAI ou modelos do HuggingFace Hub.
- **Interface Interativa:** Apresenta uma interface web desenvolvida em Streamlit, permitindo que o usuário interaja por meio de um chat simples e intuitivo.

## Tecnologias Utilizadas

- **Python 3.11+** – Linguagem de programação principal.  
- **Streamlit** – Framework web para a interface gráfica interativa.  
- **LangChain** – Biblioteca para construir pipelines de processamento de linguagem (incluindo RAG).  
- **FAISS** – Biblioteca de busca vetorial para armazenar e recuperar embeddings dos documentos.  
- **Hugging Face** – Plataforma para modelos pré-treinados; usada via HuggingFace Hub e embeddings (e.g., modelo *BAAI/bge-m3*).  
- **OpenAI API** – Acesso a modelos da família GPT (como GPT-4) via API.  
- **Ollama** – Integração com modelos de linguagem locais através do servidor Ollama.  
- **Python-dotenv** – Gerenciamento de variáveis de ambiente definidas em arquivo `.env`.  
- **Pandas** – Manipulação de dados tabulares (CSV, Excel) e formatação de texto.  

## Instalação

Para instalar o GenAI 1.0.0, siga estes passos:

1. **Criar um ambiente virtual:**  
   É recomendado criar e ativar um ambiente virtual para isolar as dependências do projeto. Por exemplo:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # No Windows use: .venv\Scripts\activate
   ```

2. **Instalar dependências:**  
   Com o ambiente ativado, instale os pacotes necessários a partir do arquivo `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   O arquivo `requirements.txt` inclui, entre outras, as principais bibliotecas do projeto, como:
   - `streamlit`  
   - `langchain` (incluindo módulos auxiliares como langchain-ollama, langchain-openai etc.)  
   - `openai`  
   - `huggingface-hub`  
   - `python-dotenv`  
   - `pandas`  
   - `faiss-cpu` (ou `faiss-gpu`)  

3. **Configurar variáveis de ambiente:**  
   Crie um arquivo `.env` na raiz do projeto para definir suas chaves de API e configurações sensíveis. Por exemplo:

   ```bash
   OPENAI_API_KEY="sua-chave-openai"
   HUGGINGFACEHUB_API_TOKEN="seu-token-huggingface"
   OLLAMA_API_KEY="seu-token-ollama"
   ```

   Ajuste conforme os serviços que você deseja utilizar. O projeto usa `python-dotenv` para carregar essas variáveis automaticamente.

## Execução

Com o ambiente configurado e as dependências instaladas, inicie a aplicação Streamlit executando o arquivo principal:

```bash
streamlit run main.py
```

Isso abrirá a interface web no seu navegador (geralmente em `http://localhost:8501`). Caso tenha configurado variáveis de ambiente no `.env`, elas já estarão carregadas. Caso contrário, certifique-se de defini-las antes da execução.

## Exemplos de Uso

Após iniciar o GenAI, a interface exibirá um painel lateral para upload de arquivos e uma área principal de chat. Para utilizar o sistema:

- Faça upload de um ou mais documentos (PDF, CSV, XLSX, XLS, TXT ou JSON) usando o seletor de arquivos na barra lateral.
- No campo de chat, digite perguntas ou comandos relacionados ao conteúdo dos documentos. Por exemplo:  
  - “Resuma o relatório do PDF enviado.”  
  - “Quais são os principais dados contidos neste arquivo CSV?”  
  - “Liste as informações-chave do arquivo JSON carregado.”
- O assistente analisará os documentos carregados usando RAG e os LLMs selecionados (OpenAI, HuggingFace ou Ollama) e retornará uma resposta contextualizada no chat.

Assim, o usuário pode interagir livremente: cada pergunta enviada refere-se ao conteúdo dos arquivos carregados, e o GenAI fornece respostas em português de forma concisa e relevante.

## Requisitos de Versão

O projeto requer Python 3.11 ou superior e as seguintes dependências (valores de versão ilustrativos):

- `streamlit >= 1.25.0`  
- `langchain >= 0.0.219`  
- `openai >= 0.27.0`  
- `huggingface-hub >= 0.15.0`  
- `python-dotenv >= 0.21.0`  
- `pandas >= 2.0.0`  
- `faiss-cpu >= 1.7.1`  

Verifique o arquivo `requirements.txt` para as versões exatas usadas no projeto. Certifique-se de instalar as versões compatíveis para evitar conflitos.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir *issues* relatando bugs ou sugerindo melhorias, assim como para enviar *pull requests* com novos recursos e correções. Algumas ideias para contribuições futuras incluem:

- Suporte a mais formatos de arquivo (imagens, XML, etc.).  
- Implementação de modelos adicionais (outros LLMs ou embeddings).  
- Otimização do desempenho do sistema de RAG (cache, paralelismo).  
- Melhorias na interface de usuário (layout, temas, usabilidade).  

Agradecemos qualquer colaboração que torne o GenAI 1.0.0 ainda mais útil e robusto para a comunidade!
