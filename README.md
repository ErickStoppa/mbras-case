# MBRAS Backend Challenge - Sentiment Analysis API

API de análise de sentimentos em tempo real com foco em:

- determinismo
- performance
- qualidade de código
- desenvolvimento orientado a testes (TDD)

## Tecnologias

- Python 3.11
- FastAPI
- Pytest
- GitHub Actions (CI)

## Como rodar o projeto

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows (Git Bash)
pip install -r requirements.txt
uvicorn main:app --reload
```

Documentação interativa:

- http://127.0.0.1:8000/docs

## Testes

Suíte completa:

```bash
python -m pytest -q tests
```

Teste de performance:

```bash
RUN_PERF=1 python -m pytest -q tests/test_performance.py --durations=1
```

Resultado atual de referência: ~30ms para 1000 mensagens.

## Endpoint

`POST /analyze-feed`

Exemplo de payload:

```json
{
  "time_window_minutes": 60,
  "messages": [
    {
      "user_id": "user_mbras_123",
      "content": "Super adorei o produto #produto",
      "timestamp": "2025-09-01T12:00:00Z",
      "hashtags": ["#produto"],
      "reactions": 7,
      "shares": 0,
      "views": 100
    }
  ]
}
```

### Determinismo

- SHA-256 para cálculo de followers
- `now_utc` injetável no domínio para testes
- mesma entrada gera mesma saída

### Separação de responsabilidades

- `sentiment_analyzer.py`: regras de domínio
- `main.py`: camada HTTP

### Performance

Otimizações aplicadas com `lru_cache` em:

- parsing de timestamp
- tokenização
- normalização
- análise de sentimento

### Janela temporal

- endpoint usa o tempo atual da requisição em UTC
- domínio mantém `now_utc` injetável para testes determinísticos

## Funcionalidades implementadas

- análise de sentimento com precedência de regras
- influence ranking com engagement real
- trending topics com peso temporal e modificador por sentimento
- detecção de anomalias

## CI

Pipeline em `.github/workflows/ci.yml` com:

- setup e instalação de dependências
- checagem simples de import/startup
- execução da suíte de testes

## Estrutura do projeto

```text
.
├── main.py
├── sentiment_analyzer.py
├── requirements.txt
├── tests/
├── examples/
└── .github/workflows/ci.yml
```

## Tempo de desenvolvimento

~3 horas

## Entrega

Repositório público contendo:

- código
- testes
- CI

## Observações

Durante o desenvolvimento foram identificadas pequenas inconsistências entre o enunciado e os testes.

Decisão adotada:
- priorizar os testes como fonte de verdade
- manter compatibilidade com a suíte fornecida