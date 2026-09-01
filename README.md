# Laboratório de detecção de objetos em imagens aéreas

Código experimental de apoio a dissertação de mestrado **"Redes Neurais Profundas Para Detecção de Pessoas em Imagens Aéreas Geradas Por Drones: Um enfoque comparativo"**, de Luiz Gustavo Santos Verissimo, Programa de Pós-graduação em Ciência da Computação, Instituto de Informática, Universidade Federal de Goiás (UFG), Goiânia, Goiás, Brasil.

O repositório contém uma aplicação Tkinter para normalização de datasets, treinamento, inferência, avaliação pós-treinamento, benchmark de inferência e leitura de metadados de checkpoints. A prioridade do projeto e reprodutibilidade experimental, rastreabilidade e transparência sobre limitações conhecidas.

## Escopo científico

Modelos comparados:

- YOLOv12n, via Ultralytics.
- SSD300, via `torchvision.models.detection.ssd300_vgg16`.
- Faster R-CNN, via `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
- RetinaNet, via `torchvision.models.detection.retinanet_resnet50_fpn`.

Datasets principais:

- HERIDAL: copia Kaggle com 1.546 imagens, dividida de forma reprodutível em 1.236 treino e 310 validacao, `seed=42`.
- VisDrone2019-DET: validação oficial com 548 imagens, 38.759 instancias e 10 categorias.

Os datasets e checkpoints não devem ser versionados neste repositório. Consulte as licenças e páginas oficiais antes de redistribuir dados ou pesos treinados.

## Estado do código

A versão publica foi auditada para remover problemas que comprometeriam novas execuções:

- todos os campos de `TrainConfig` sao expostos como hiperparâmetros na GUI;
- filtros por classe foram removidos dos fluxos de inferência/normalização;
- o leitor VOC/SSD atual preserva as classes existentes;
- anotações de validação são obrigatórias em processos COCO de validação;
- testes de regressão cobrem preservação multiclasse do VisDrone.

Essas correções são posteriores a parte dos experimentos históricos. Portanto, resultados já reportados na dissertação não devem ser reinterpretados como se tivessem sido produzidos pelo código corrigido atual.

## Documentos de leitura obrigatória

- `REPRODUCIBILITY.md`: protocolo histórico, ambiente, datasets, checkpoints, métricas e diferenças entre código histórico e atual.
- `KNOWN_ISSUES.md`: limitações conhecidas, incluindo o problema histórico SSD300/VisDrone.
- `AUDIT_REPORT.md`: relatório de auditoria técnica e cientifica.
- `SHA256SUMS.txt`: hashes SHA-256 dos checkpoints históricos localizados em `C:\Experimentos`.
- `SECURITY.md`: política de segurança e orientações para reportar problemas.

## Estrutura

- `app/gui.py`: interface gráfica e coleta de parâmetros.
- `app/controller.py`: orquestração dos pipelines.
- `app/datasets/`: representação intermediaria, readers e exporters YOLO/VOC/COCO.
- `app/detectors/`: modelos, treino, validação, inferência e utilitários.
- `app/avaliacao/`: métricas COCO, métricas operacionais e figuras.
- `app/training/`: checkpointing e early stopping.
- `tests/`: testes rápidos de regressão cientifica e smoke tests.
- `tools/`: scripts auxiliares de auditoria; podem exigir ajuste de caminhos locais antes de uso.

## Instalação

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

As versões em `requirements.txt` refletem dependências conhecidas da aplicação publica. Para reprodução histórica estrita, use também as versões e ressalvas em `REPRODUCIBILITY.md`.

## Execução

```bash
python -m app.main
```

Na GUI, selecione algoritmo, acao e caminhos. Para reproduzir uma execução histórica, informe explicitamente os hiperparâmetros do respectivo experimento; não assuma que os defaults atuais representam todos os treinos reportados na dissertação.

## Validação e métricas

As avaliações COCO exigem anotações de validação explicitas. O código atual evita inferência automática silenciosa de `instances_val.json`.

AP/mAP/AR usam avaliacao COCO e preservam predições de baixa confiança para ranking. Precision/Recall/F1 e matriz de confusão são calculadas separadamente no ponto operacional documentado.

## Testes

```bash
pytest -q
```

Os testes não executam treinamento completo. Eles cobrem parsing de datasets, preservação multiclasse do VisDrone, convenções de classes, métricas sintéticas, checkpointing rápido e consistência básica dos parâmetros de treino.

## Licença

O código deste repositório está sob licenca MIT, conforme `LICENSE`. Dependências e datasets possuem licenças próprias; em especial, confira as condições de uso do Ultralytics e dos datasets antes de redistribuir artefatos.
