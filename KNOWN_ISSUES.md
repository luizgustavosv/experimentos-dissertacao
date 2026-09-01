# Problemas conhecidos

## SSD300/VisDrone historico reteve apenas `pedestrian`

Severidade: CRITICAL.

A dissertacao informa um defeito historico no pipeline SSD300/VisDrone: o leitor de anotacoes usado no treinamento reteve exclusivamente a categoria `pedestrian`. A versao atual foi corrigida para preservar as dez categorias do VisDrone, e ha teste automatizado para impedir regressao.

Impacto: resultados historicos do par SSD300/VisDrone nao devem ser reinterpretados como avaliacao multiclasse completa. Se a dissertacao ja registra esse defeito, a correcao atual deve ser descrita como posterior ao experimento.

## Confronto com a dissertacao pendente

O arquivo da dissertacao nao estava presente no workspace durante esta auditoria. Assim, a comparacao sistematica entre capitulos de Materiais e Metodos/Resultados e o codigo ainda precisa ser concluida quando o PDF/DOC for disponibilizado.

## Caminhos de dados locais em scripts auxiliares

Scripts auxiliares que dependem de datasets ou historicos fora do repositorio exigem argumentos CLI ou variaveis de ambiente, como `EXPERIMENTOS_CNNS_ROOT`, `HERIDAL_VAL_COCO` e `VISDRONE_VAL_COCO`. Isso evita publicar caminhos de maquina local como se fossem parte do protocolo experimental.

## Versoes historicas de dependencias

O repositorio nao contem evidencia suficiente para fixar todas as versoes exatas usadas nos experimentos historicos. Atualizar dependencias sem essa evidencia poderia criar falsa reprodutibilidade.

## Benchmark de velocidade

Os fluxos de inferencia medem tempo com sincronizacao CUDA antes/depois do loop, mas nao foi implementado warm-up padronizado nem protocolo unico por modelo. Comparacoes de FPS devem registrar hardware, checkpoint, split e protocolo exato.
