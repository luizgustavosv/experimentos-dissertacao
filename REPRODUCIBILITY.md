# Reprodutibilidade

## Rastreabilidade

- Branch de auditoria/publicacao: `audit-publicacao-cientifica`.
- Commit base observado antes desta rodada de auditoria: `20b18aae0665f62374b3e1c427b9359fcb5c5b64`.
- O worktree ja continha alteracoes nao commitadas antes desta etapa; elas nao foram revertidas nem reescritas.

## Ambiente

O projeto usa Python, PyTorch, Torchvision, Ultralytics, pycocotools, NumPy, Pillow, OpenCV headless, Matplotlib, fpdf2, PyYAML e pytest.

As versoes exatas usadas em todos os experimentos historicos nao estao inequivocamente registradas no repositorio. Portanto, nao foi fabricado um lockfile retroativo. Para reproducao nova, crie um ambiente limpo e instale `requirements.txt`.

## Seeds e determinismo

- `TrainConfig.seed` e exposto na GUI.
- O treino Torchvision chama `seed_everything(config.seed)`.
- DataLoaders de treino usam `shuffle=True`; validacao usa `shuffle=False`.
- Determinismo completo de CUDA/cuDNN nao esta garantido pelo codigo atual.

## Dados

Datasets nao sao versionados. Prepare HERIDAL e VisDrone localmente e use a GUI para normalizar para:

- YOLO: `dataset.yaml`, `images/`, `labels/`.
- SSD300: Pascal VOC (`JPEGImages`, `Annotations`, `ImageSets/Main`, `labels.txt`).
- Faster R-CNN/RetinaNet: COCO (`images/train`, `images/val`, JSONs de anotacao).

Para qualquer treino ou validacao COCO, o JSON de validacao deve ser informado explicitamente.

## Versao historica versus versao corrigida

O codigo atual contem correcoes posteriores aos experimentos historicos, principalmente:

- remocao de filtros por classe na GUI/inferencia;
- preservacao multiclasse em readers/normalizadores;
- obrigatoriedade de anotacoes de validacao;
- exposicao integral dos hiperparametros de `TrainConfig`.

Essas correcoes melhoram a versao publica do laboratorio, mas nao alteram automaticamente os resultados ja reportados. Consulte `AUDIT_REPORT.md` e `KNOWN_ISSUES.md`.

## Validacao local

```bash
python -m py_compile app/gui.py app/controller.py app/detectors/base.py app/detectors/utils.py app/detectors/torchvision_detectors.py app/detectors/ssd.py app/detectors/yolo.py app/datasets/readers/heridal_reader.py app/datasets/normalizer.py normalize_dataset.py
pytest -q
```
