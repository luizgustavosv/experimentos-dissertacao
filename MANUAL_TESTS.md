# Testes manuais planejados

- Caso 1: Selecionar o algoritmo Faster R-CNN e abrir a Action Box → o botão/ação "Validar" aparece.
- Caso 2: Escolher pesos do experimento (ex.: `last.pth`) para Faster R-CNN → a validação roda e os logs mostram as chaves de loss do torchvision (`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`).
- Caso 3: Selecionar um checkpoint global (por exemplo `C:\Experimentos\CNNs\last.pth`) → os pesos são carregados corretamente e a validação é executada.
- Caso 4: Escolher um arquivo de pesos inválido ou muito pequeno → a execução é bloqueada com mensagem clara, sem crash.
