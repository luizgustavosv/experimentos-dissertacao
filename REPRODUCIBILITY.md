# Reprodutibilidade

Este documento sincroniza o repositorio com a versao final da dissertacao preservada em `C:\Experimentos\Dissertacao_corrigida.zip`, especialmente `tex/cap_III.tex`, `tex/cap_IV.tex`, `tex/cap_V.tex` e `CHANGELOG_CORRECOES.md`.

## Escopo

O codigo atual e uma versao corrigida para publicacao. Ele nao e tratado como copia byte-a-byte do codigo usado em todos os treinos historicos. Quando ha diferenca entre comportamento historico e comportamento atual, a diferenca e documentada.

## Ambiente computacional historico

Plataforma fisica reportada:

- Dell Inspiron 15.
- Windows 11 Home 64-bit.
- Intel Core i7-11390H @ 3.40 GHz.
- 24 GB RAM.
- SSD 500 GB.
- NVIDIA GeForce MX450 2 GB.

Registros preservados indicam Python 3.12.0, PyTorch 2.2.2+cpu, Torchvision 0.17.2 e Ultralytics 8.3.162 em execucoes documentadas. Treinamentos e avaliacoes preditivas preservadas foram executados em CPU; o benchmark final de velocidade foi executado na GPU MX450.

## Datasets

| Dataset | Uso historico | Classes | Observacoes |
|---|---:|---:|---|
| HERIDAL oficial | 1.647 imagens anotadas; 1.546 treino; 101 teste; 3.229 instancias de pessoas | 1 | A dissertacao usou copia Kaggle com 1.546 imagens. |
| HERIDAL experimental | 1.236 treino; 310 validacao; `seed=42` | 1 | Split customizado aproximado 80/20; validacao final com 685 instancias humanas. |
| VisDrone2019-DET validacao | 548 imagens; 38.759 instancias | 10 | Validacao oficial; categorias originais preservadas. |

No VisDrone, as categorias `pedestrian` e `people` foram analisadas separadamente. Nao foi criado um rotulo artificial `human` para fundir essas categorias.

## Configuracao de treino historica

| Familia | Batch | Orcamento maximo | Otimizador descrito na dissertacao | Scheduler |
|---|---:|---:|---|---|
| YOLOv12n | 16 | 100 epocas | AdamW | Gerenciado pelo Ultralytics |
| SSD300 | 2 | 100 epocas | SGD | StepLR, `step_size=3`, `gamma=0.1` |
| Faster R-CNN | 2 | 100 epocas | SGD | StepLR, `step_size=3`, `gamma=0.1` |
| RetinaNet | 2 | 100 epocas | SGD | StepLR, `step_size=3`, `gamma=0.1` |

O monitoramento interno de early stopping usou 10% da particao de treino. A validacao final reportada usa os splits de validacao descritos acima.

## Early stopping

| Modelo | Patience historico documentado | Observacao |
|---|---:|---|
| SSD300 | 10 | Alinhado aos artefatos `args.yaml` localizados. |
| YOLOv12n | 25 | A dissertacao documenta 25; artefatos YOLO preservados registram `patience: 100` em alguns `args.yaml` principais e `patience: 10` em runs auxiliares de 20 epocas. |
| Faster R-CNN | 25 | Artefatos de segmentos retomados registram variacoes, incluindo `early_stop_patience: 20` no terceiro segmento VisDrone. |
| RetinaNet | 25 | Alinhado aos artefatos principais localizados. |

Para novas execucoes, configure `early_stop_patience` explicitamente na GUI. O default atual do codigo nao deve ser usado como evidencia historica.

## Checkpoints historicos

| Modelo | Dataset | Checkpoint reportado | Epocas/criterio | Politica de selecao historica |
|---|---|---|---|---|
| YOLOv12n | HERIDAL | `best.pt` | treino chegou a 100 epocas; epoca interna do melhor nao recuperada | melhor preservado pelo Ultralytics |
| YOLOv12n | VisDrone | `best.pt` | 100 epocas acumuladas, 94+6; melhor no segmento final retomado | melhor preservado pelo Ultralytics apos retomada de `last.pt` |
| SSD300 | HERIDAL | `best_by_monitor.pth` | epoca 100; `early_stopping.best_epoch=100` | melhor por monitoramento interno |
| SSD300 | VisDrone | `ssd_last_epoch_0100.pth` | epoca 100; `val_loss=5.3980`; `best.pth` antigo da epoca 10 tinha `val_loss=5.5372` | ultimo checkpoint valido preservado; `best_path` final nao foi preservado |
| Faster R-CNN | HERIDAL | `checkpoint_epoch_65.pth` | 100 epocas acumuladas, 35+65 | ultimo checkpoint fisicamente disponivel do segmento retomado |
| Faster R-CNN | VisDrone | `checkpoint_epoch_49.pth` | 82 epocas acumuladas, 5+28+49 | ultimo checkpoint fisicamente disponivel do segmento retomado |
| RetinaNet | HERIDAL | `checkpoint_epoch_100.pth` | epoca 100 | ultimo/final checkpoint disponivel |
| RetinaNet | VisDrone | `checkpoint_epoch_48.pth` | epoca 48; metadado aponta `best_epoch=8` | melhor da epoca 8 nao localizado fisicamente; usado ultimo checkpoint disponivel |

Hashes SHA-256 dos arquivos localizados estao em `SHA256SUMS.txt`.

## Metricas

| Grupo de metricas | Parametros historicos |
|---|---|
| Precision/Recall/F1 e matriz de confusao | confianca 0.25; IoU 0.5 |
| AP/AR COCO | predicoes exportadas com confianca 0.001 |
| HERIDAL COCO | `maxDets=[1, 10, 100]` |
| VisDrone COCO | `maxDets=[1, 10, 100, 500]` |
| Diagnostico expandido | `maxDets=5000` |

Tetos de exportacao historicos:

| Modelo | Entrada | Conf. operacional | Conf. exportacao | Teto de predicoes |
|---|---:|---:|---:|---:|
| YOLOv12n | 640x640 | 0.25 | 0.001 | 5000 |
| SSD300 | 300x300 | 0.25 | 0.001 | 5000 |
| Faster R-CNN | 640x640 + transform interno | 0.25 | 0.001 | 10000 |
| RetinaNet | 640x640 + transform interno | 0.25 | 0.001 | 5000 |

## Benchmark de velocidade

O benchmark final foi feito na validacao HERIDAL, com 310 imagens, checkpoints treinados no HERIDAL, plataforma local e GPU NVIDIA GeForce MX450 2 GB. O modo foi inferencia rapida, sem renderizacao nem salvamento de caixas. A medida representa latencia end-to-end das rotinas implementadas, nao apenas tempo isolado de forward.

Resultados reportados na dissertacao:

| Modelo | FPS | Latencia aproximada |
|---|---:|---:|
| YOLOv12n | 6.07 | 164.84 ms |
| SSD300 | 1.76 | 568.18 ms |
| RetinaNet | 1.14 | 877.19 ms |
| Faster R-CNN | 1.02 | 980.39 ms |

## Problema SSD300/VisDrone

O treinamento historico SSD300/VisDrone foi afetado por defeito no leitor VOC: apenas a classe `pedestrian` foi retida como positiva, apesar de metadados declararem dez categorias. O conjunto de validacao de referencia permanecia multiclasse e intacto. Por isso, as metricas agregadas de dez classes desse par sao mantidas apenas como registro historico e nao devem ser usadas como comparacao multiclasse plena.

Metricas historicas SSD300/VisDrone documentadas: precision micro 0.0102, recall micro 0.0368, F1 0.0160, mAP@0.5 0.0016, mAP@[0.5:0.95] 0.0004 e AP@0.5 para `pedestrian` 0.0146.

O codigo atual foi corrigido para preservar classes. Reexecutar SSD300/VisDrone com o leitor atual gera um novo experimento, nao uma correcao retroativa dos resultados historicos.

## Otimizador YOLO

A dissertacao descreve YOLOv12n com AdamW. Os artefatos locais preservados nao confirmam isso de forma inequivoca:

- `C:\Experimentos\CNNs\YOLO\yolo12n_visdrone\pesos\yolo_visdrone*\args.yaml` registra `optimizer: auto`, `lr0: 0.01`, `momentum: 0.937`, `weight_decay: 0.0005`.
- runs auxiliares de 20 epocas tambem registram `optimizer: auto`.
- `app.log` contem ao menos uma execucao YOLO em que o Ultralytics 8.3.162 resolveu `optimizer=auto` para `SGD(lr=0.01, momentum=0.9)`.

Nao foi encontrada evidencia local suficiente para alterar o codigo atual e forcar AdamW como reproducao historica. Esta divergencia deve permanecer documentada ate revisao humana dos logs completos ou de artefatos adicionais.

## Validacao local do repositorio

```bash
python -m py_compile app/gui.py app/controller.py app/detectors/base.py app/detectors/utils.py app/detectors/torchvision_detectors.py app/detectors/ssd.py app/detectors/yolo.py app/datasets/readers/heridal_reader.py app/datasets/normalizer.py normalize_dataset.py
pytest -q
```

