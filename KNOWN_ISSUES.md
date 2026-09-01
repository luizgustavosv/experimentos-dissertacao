# Problemas conhecidos

## SSD300/VisDrone historico reteve apenas `pedestrian`

Severidade: CRITICAL.

A dissertacao registra um defeito historico no pipeline SSD300/VisDrone: o leitor de anotacoes usado no treinamento reteve exclusivamente a categoria `pedestrian`. A versao atual foi corrigida para preservar as dez categorias do VisDrone, e ha teste automatizado para impedir regressao.

Impacto: resultados historicos do par SSD300/VisDrone nao representam avaliacao multiclasse completa. As metricas agregadas em dez classes devem ser lidas como registro historico do experimento afetado; a comparacao cientificamente defensavel para esse par e restrita ao escopo `pedestrian`, conforme explicitado na dissertacao.

## Otimizador historico do YOLO nao confirmado pelos artefatos locais

Severidade: HIGH.

A dissertacao descreve YOLOv12n com AdamW. Os `args.yaml` preservados localmente para os treinos YOLO principais registram `optimizer: auto`, nao `AdamW`. O `app.log` tambem contem ao menos uma execucao em que o Ultralytics 8.3.162 resolveu `optimizer=auto` para SGD.

Impacto: nao ha base documental suficiente para afirmar, no repositorio, que os checkpoints YOLO historicos foram treinados com AdamW. A documentacao publica registra a divergencia; o codigo atual nao foi alterado para forcar AdamW retroativamente.

## Codigo atual corrigido versus codigo historico

Severidade: HIGH.

A versao publica remove filtros por classe, preserva classes em readers/normalizadores e exige anotacoes de validacao explicitas. Essas correcoes melhoram a qualidade cientifica de novas execucoes, mas nao recriam exatamente todos os experimentos historicos.

Impacto: ao reproduzir a dissertacao, use os metadados e checkpoints historicos documentados em `REPRODUCIBILITY.md`. Ao executar novos treinos, trate os resultados como nova rodada experimental.

## Patience historico e artefatos retomados

Severidade: MEDIUM.

A dissertacao documenta early stopping com patience 10 para SSD300 e 25 para YOLOv12n, Faster R-CNN e RetinaNet. Artefatos locais de execucoes retomadas registram variacoes em alguns segmentos, como `patience: 100` em `args.yaml` do YOLO e `early_stop_patience: 20` no terceiro segmento Faster R-CNN/VisDrone.

Impacto: o repositorio documenta a politica historica da dissertacao e tambem alerta que os artefatos de segmentos retomados podem conter valores parciais. Para novas execucoes, configure explicitamente o valor desejado.

## Caminhos de dados locais em scripts auxiliares

Severidade: MEDIUM.

Scripts auxiliares que dependem de datasets ou historicos fora do repositorio exigem argumentos CLI ou variaveis de ambiente, como `EXPERIMENTOS_CNNS_ROOT`, `HERIDAL_VAL_COCO` e `VISDRONE_VAL_COCO`. Isso evita publicar caminhos de maquina local como se fossem parte do protocolo experimental.

## Versoes historicas de dependencias

Severidade: MEDIUM.

O repositorio nao contem evidencia suficiente para fixar todas as versoes exatas usadas em todos os experimentos historicos. Atualizar dependencias ou fabricar um lockfile retroativo criaria falsa reprodutibilidade. As versoes recuperadas estao em `REPRODUCIBILITY.md`.

## Benchmark de velocidade

Severidade: LOW.

O benchmark final reportado na dissertacao mede latencia end-to-end das rotinas implementadas em GPU MX450, sem renderizacao nem salvamento de caixas. Ele nao deve ser reinterpretado como tempo isolado de forward do modelo.

