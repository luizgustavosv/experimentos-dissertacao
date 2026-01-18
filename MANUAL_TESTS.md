# Testes manuais planejados

- Caso 1: Selecionar o algoritmo Faster R-CNN e abrir a Action Box → o botão/ação "Validar" aparece.
- Caso 2: Escolher pesos do experimento (ex.: `last.pth`) para Faster R-CNN → a validação roda e os logs mostram as chaves de loss do torchvision (`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`).
- Caso 3: Selecionar um checkpoint global (por exemplo `C:\Experimentos\CNNs\last.pth`) → os pesos são carregados corretamente e a validação é executada.
- Caso 4: Escolher um arquivo de pesos inválido ou muito pequeno → a execução é bloqueada com mensagem clara, sem crash.
- Caso 5: Early stopping DESATIVADO (qualquer detector) → treino percorre todas as épocas e args.yaml registra `early_stopping.enabled: false`.
- Caso 6: Early stopping ATIVADO com patience=2 (SSD/Faster R-CNN/RetinaNet) → treino para antes do total de épocas quando a loss de validação não melhora; logs exibem monitor/valor e args.yaml salva patience=2.
- Caso 7: Early stopping ATIVADO com patience=2 (YOLO) → args.yaml do run do Ultralytics registra `patience: 2` e logs mostram early_stopping_enabled/patience no início do treino.
