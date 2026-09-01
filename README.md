# Laboratorio de deteccao de objetos em imagens aereas

Codigo experimental de apoio a dissertacao de mestrado **"Redes Neurais Profundas Para Deteccao de Pessoas em Imagens Aereas Geradas Por Drones: Um enfoque comparativo"**, de Luiz Gustavo Santos Verissimo, Programa de Pos-Graduacao em Ciencia da Computacao, Instituto de Informatica, Universidade Federal de Goias (UFG), Goiania, Goias, Brasil.

O repositorio contem uma aplicacao Tkinter para normalizacao de datasets, treinamento, inferencia, avaliacao pos-treinamento, benchmark de inferencia e leitura de metadados de checkpoints. A prioridade do projeto e reprodutibilidade experimental, rastreabilidade e transparencia sobre limitacoes conhecidas.

## Escopo cientifico

Modelos comparados:

- YOLOv12n, via Ultralytics.
- SSD300, via `torchvision.models.detection.ssd300_vgg16`.
- Faster R-CNN, via `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
- RetinaNet, via `torchvision.models.detection.retinanet_resnet50_fpn`.

Datasets principais:

- HERIDAL: copia Kaggle com 1.546 imagens, dividida de forma reprodutivel em 1.236 treino e 310 validacao, `seed=42`.
- VisDrone2019-DET: validacao oficial com 548 imagens, 38.759 instancias e 10 categorias.

Os datasets e checkpoints nao devem ser versionados neste repositorio. Consulte as licencas e paginas oficiais antes de redistribuir dados ou pesos treinados.

## Estado do codigo

A versao publica foi auditada para remover problemas que comprometeriam novas execucoes:

- todos os campos de `TrainConfig` sao expostos como hiperparametros na GUI;
- filtros por classe foram removidos dos fluxos de inferencia/normalizacao;
- o leitor VOC/SSD atual preserva as classes existentes;
- anotacoes de validacao sao obrigatorias em processos COCO de validacao;
- testes de regressao cobrem preservacao multiclasse do VisDrone.

Essas correcoes sao posteriores a parte dos experimentos historicos. Portanto, resultados ja reportados na dissertacao nao devem ser reinterpretados como se tivessem sido produzidos pelo codigo corrigido atual.

## Documentos de leitura obrigatoria

- `REPRODUCIBILITY.md`: protocolo historico, ambiente, datasets, checkpoints, metricas e diferencas entre codigo historico e atual.
- `KNOWN_ISSUES.md`: limitacoes conhecidas, incluindo o problema historico SSD300/VisDrone.
- `AUDIT_REPORT.md`: relatorio de auditoria tecnica e cientifica.
- `SHA256SUMS.txt`: hashes SHA-256 dos checkpoints historicos localizados em `C:\Experimentos`.
- `SECURITY.md`: politica de seguranca e orientacoes para reportar problemas.

## Estrutura

- `app/gui.py`: interface grafica e coleta de parametros.
- `app/controller.py`: orquestracao dos pipelines.
- `app/datasets/`: representacao intermediaria, readers e exporters YOLO/VOC/COCO.
- `app/detectors/`: modelos, treino, validacao, inferencia e utilitarios.
- `app/avaliacao/`: metricas COCO, metricas operacionais e figuras.
- `app/training/`: checkpointing e early stopping.
- `tests/`: testes rapidos de regressao cientifica e smoke tests.
- `tools/`: scripts auxiliares de auditoria; podem exigir ajuste de caminhos locais antes de uso.

## Instalacao

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

As versoes em `requirements.txt` refletem dependencias conhecidas da aplicacao publica. Para reproducao historica estrita, use tambem as versoes e ressalvas em `REPRODUCIBILITY.md`.

## Execucao

```bash
python -m app.main
```

Na GUI, selecione algoritmo, acao e caminhos. Para reproduzir uma execucao historica, informe explicitamente os hiperparametros do respectivo experimento; nao assuma que os defaults atuais representam todos os treinos reportados na dissertacao.

## Validacao e metricas

As avaliacoes COCO exigem anotacoes de validacao explicitas. O codigo atual evita inferencia automatica silenciosa de `instances_val.json`.

AP/mAP/AR usam avaliacao COCO e preservam predicoes de baixa confianca para ranking. Precision/Recall/F1 e matriz de confusao sao calculadas separadamente no ponto operacional documentado.

## Testes

```bash
pytest -q
```

Os testes nao executam treinamento completo. Eles cobrem parsing de datasets, preservacao multiclasse do VisDrone, convencoes de classes, metricas sinteticas, checkpointing rapido e consistencia basica dos parametros de treino.

## Licenca

O codigo deste repositorio esta sob licenca MIT, conforme `LICENSE`. Dependencias e datasets possuem licencas proprias; em especial, confira as condicoes de uso do Ultralytics e dos datasets antes de redistribuir artefatos.
