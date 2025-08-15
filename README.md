# Imagem ⇄ Matriz (tons de cinza) — Flask + Docker

## Rodar
```bash
docker compose build
docker compose up
# abrir http://localhost:8000
```

## Funcionalidades
- Converter imagem → matriz (texto).
- Recriar imagem ← matriz (PNG).
- Métricas: largura, altura, min, max, média, desvio padrão, **entropia**, **tempo de processamento**, **tamanho do arquivo**.
- Histograma de tons de cinza.

## Formato da matriz
```
<w> <h>
p00 p01 ... p0,w-1
...
p(h-1),0 ... p(h-1),w-1
```
Valores 0–255.


## Visão Geral

Este projeto é uma aplicação web desenvolvida em **Flask** para converter imagens em matrizes de tons de cinza (formato texto) e reconstruir imagens a partir dessas matrizes. Ele oferece métricas estatísticas e visualização do histograma. O deploy é feito via **Docker** e **Docker Compose**.

---

## Componentes e Utilização

### 1. **app/**

- **main.py**  
  Arquivo principal da aplicação Flask. Define rotas:
  - `/` — Página inicial, upload de imagem ou matriz.
  - `/process` — Processa uploads (encode/decode).
  - `/output/<filename>` — Serve arquivos gerados (imagem, matriz, histograma).  
  Utiliza variáveis de ambiente para o segredo do Flask.

- **processing.py**  
  Funções para conversão entre imagem ↔ matriz, cálculo de métricas, geração de histograma.  
  Utiliza **Pillow** para manipulação de imagens.

- **style.css**  
  Estilos para a interface web.

- **index.html**  
  Página inicial, formulário para upload de imagem ou matriz.

- **result.html**  
  Página de resultado, mostra pré-visualização, métricas, histograma e links para download.

### 2. **output/**  
Diretório para arquivos gerados (matrizes, imagens reconstruídas, histogramas).

### 3. **uploads/**  
Diretório para uploads temporários do usuário.

### 4. **requirements.txt**  
Dependências Python:
- **Flask** — Framework web.
- **Pillow** — Manipulação de imagens.
- **gunicorn** — Servidor WSGI para produção.

### 5. **Dockerfile**  
Define o ambiente do container:
- Base: `python:3.11-slim`
- Instala dependências do Pillow (libjpeg, zlib).
- Instala dependências Python.
- Expõe porta 8000.
- Executa com Gunicorn.

### 6. **docker-compose.yml**  
Orquestra o serviço web:
- Build da imagem.
- Mapeia porta 8000.
- Define variáveis de ambiente.
- Monta volumes para persistência de arquivos gerados e uploads.

### 7. **README.md**  
Instruções de uso, funcionalidades e formato da matriz.

---

## Por que cada tecnologia está sendo utilizada?

- **Flask**: Framework leve e fácil para APIs web.
- **Pillow**: Manipulação de imagens (leitura, conversão, geração de histogramas).
- **Gunicorn**: Servidor WSGI robusto para produção.
- **Docker/Docker Compose**: Facilita o deploy, isolamento e portabilidade do ambiente.
- **HTML/CSS**: Interface amigável para o usuário.
- **Volumes Docker**: Persistência dos arquivos gerados e uploads fora do container.

---

## Como rodar o projeto

```sh
docker compose build
docker compose up
# Acesse http://localhost:8000
