# Laboratorio de deteccao de objetos em imagens aereas

Codigo experimental de apoio a uma dissertacao de Mestrado em Ciencia da Computacao sobre deteccao de pessoas em imagens aereas geradas por drones, com comparacao entre YOLOv12n, SSD300, Faster R-CNN e RetinaNet nos datasets HERIDAL e VisDrone.

O repositorio contem uma aplicacao Tkinter para normalizacao de datasets, treinamento, inferencia, avaliacao pos-treinamento, benchmark de inferencia e leitura de metadados de checkpoints. A prioridade do projeto e reprodutibilidade experimental, rastreabilidade e transparencia sobre limitacoes conhecidas.

Autor: Luiz Gustavo.
Instituicao/programa: preencher antes da publicacao, conforme a versao final da dissertacao.
Licenca: pendente de escolha explicita. Sem um arquivo `LICENSE`, o codigo fica publicamente visivel, mas sem permissao clara de reutilizacao por terceiros.

## Modelos

- YOLO: backend Ultralytics.
- SSD300: `torchvision.models.detection.ssd300_vgg16`.
- Faster R-CNN: `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
- RetinaNet: `torchvision.models.detection.retinanet_resnet50_fpn`.

## Datasets

- HERIDAL: esperado em estrutura com `train/_annotations.csv` ou `train/annotations.csv`, conforme o fluxo usado.
- VisDrone: esperado em splits `VisDrone2019-DET-train`, `VisDrone2019-DET-val` e, quando aplicavel, `VisDrone2019-DET-test-*`, com subpastas `images/` e `annotations/`.

Datasets e checkpoints nao devem ser versionados neste repositorio. Consulte as licencas e paginas oficiais de cada dataset antes de redistribuir dados.

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

As versoes em `requirements.txt` refletem dependencias conhecidas, mas nem todas as versoes historicas foram recuperadas. Para reproducao estrita, veja `REPRODUCIBILITY.md`.

## Execucao

```bash
python -m app.main
```

Na GUI, selecione algoritmo, acao e caminhos. Todos os campos do `TrainConfig` sao expostos como hiperparametros de treinamento.

## Validacao e metricas

As avaliacoes COCO exigem anotacoes de validacao explicitas. O codigo atual evita inferencia automatica silenciosa de `instances_val.json`.

As metricas integradas de AP/mAP/AR usam avaliacao COCO e preservam predicoes de baixa confianca para ranking. Precision/Recall/F1 e matriz de confusao sao calculadas separadamente em um ponto operacional de confianca.

## Testes

```bash
pytest -q
```

Os testes nao executam treinamento completo. Eles cobrem parsing de datasets, preservacao multiclasse do VisDrone, convencoes de classes, metricas sinteticas e checkpointing rapido.

## Limitacoes conhecidas

Leia `KNOWN_ISSUES.md` antes de interpretar resultados. Em particular, ha registro de defeito historico no pipeline SSD300/VisDrone: o leitor usado no treinamento historico reteve apenas a categoria `pedestrian`. O codigo atual foi corrigido para comportamento multiclasse, mas resultados historicos nao devem ser reinterpretados como se tivessem sido gerados com o leitor corrigido.

Luiz Gustavo Santos Veríssimo
Programa de Pós-Graduação em Ciência da Computação
Instituto de Informática
Universidade Federal de Goiás (UFG)
