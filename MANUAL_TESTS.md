# Testes manuais planejados

- Caso 1: Selecionar o algoritmo Faster R-CNN e abrir a Action Box -> o botao/acao "Validar" aparece.
- Caso 2: Escolher pesos do experimento, por exemplo `last.pth`, para Faster R-CNN -> a validacao roda e os logs mostram as chaves de loss do torchvision (`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`).
- Caso 3: Selecionar um checkpoint global, por exemplo `<RAIZ_EXPERIMENTOS>\last.pth` -> os pesos sao carregados corretamente e a validacao e executada.
- Caso 4: Escolher um arquivo de pesos invalido ou muito pequeno -> a execucao e bloqueada com mensagem clara, sem crash.
- Caso 5: Early stopping DESATIVADO, qualquer detector -> treino percorre todas as epocas e `args.yaml` registra `early_stopping.enabled: false`.
- Caso 6: Early stopping ATIVADO com `patience=2`, SSD/Faster R-CNN/RetinaNet -> treino para antes do total de epocas quando a loss de validacao nao melhora; logs exibem monitor/valor e `args.yaml` salva `patience=2`.
- Caso 7: Early stopping ATIVADO com `patience=2`, YOLO -> `args.yaml` do run do Ultralytics registra `patience: 2` e logs mostram `early_stopping_enabled`/`patience` no inicio do treino.
