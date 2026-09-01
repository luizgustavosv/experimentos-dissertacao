# Audit Report

Contexto: auditoria para publicacao do codigo experimental da dissertacao. Branch observado: `audit-publicacao-cientifica`. A comparacao final usou o estado atual do repositorio e a dissertacao zipada em `C:\Experimentos\Dissertacao_corrigida.zip`.

## Fontes confrontadas

- Codigo e documentacao do repositorio atual.
- `README.md`, `KNOWN_ISSUES.md`, `REPRODUCIBILITY.md`, `AUDIT_REPORT.md` e `CITATION.cff`.
- Dissertacao final: `tex/cap_III.tex`, `tex/cap_IV.tex`, `tex/cap_V.tex`, `CHANGELOG_CORRECOES.md`.
- Artefatos locais em `C:\Experimentos\CNNs` e logs locais quando necessarios para conferir checkpoints e otimizadores.

## Achados principais

| Severidade | Problema | Impacto cientifico potencial | Correcao/acao no repositorio | Altera resultados historicos? |
|---|---|---|---|---|
| CRITICAL | SSD300/VisDrone historico reteve apenas `pedestrian`. | O resultado nao representa detector multiclasse completo. | Documentado em `KNOWN_ISSUES.md` e `REPRODUCIBILITY.md`; codigo atual preserva classes. | Nao. |
| HIGH | Dissertacao descreve YOLOv12n com AdamW, mas artefatos locais registram `optimizer: auto`; log local mostra ao menos uma resolucao para SGD. | Risco de declarar otimizador historico sem evidencia suficiente. | Divergencia documentada; codigo nao foi alterado para forcar AdamW retroativamente. | Nao. |
| HIGH | Hiperparametros reais de treino nao estavam todos selecionaveis/propagados pela GUI em versoes anteriores. | Reproducibilidade incompleta para novas execucoes. | Todos os campos de `TrainConfig` foram expostos na GUI. | Nao para o historico; sim para novas execucoes se parametros mudarem. |
| HIGH | Validacao COCO podia inferir anotacoes automaticamente. | Risco de avaliar split errado. | Anotacoes de validacao agora sao obrigatorias. | Nao. |
| HIGH | Filtros historicos por classe podiam restringir inferencias. | Comparacoes poderiam misturar single-class e multiclasse. | Opcao/filtro removidos; readers preservam classes. | Nao para o historico; novas inferencias podem diferir. |
| MEDIUM | Documentacao publica estava incompleta quanto a datasets, checkpoints, ambiente, metricas e politica de pesos. | Banca e terceiros poderiam interpretar o repositorio como mais reprodutivel do que era. | `README.md`, `REPRODUCIBILITY.md`, `KNOWN_ISSUES.md` e `CITATION.cff` sincronizados. | Nao. |
| MEDIUM | `CITATION.cff` tinha marcadores de conflito Git. | Arquivo de citacao invalido e improprio para publicacao. | Marcadores removidos e identidade academica corrigida. | Nao. |
| LOW | Benchmark de velocidade poderia ser confundido com tempo de forward isolado. | Interpretacao indevida de FPS/latencia. | Protocolo documentado como latencia end-to-end em GPU MX450. | Nao. |

## Sincronia com a dissertacao

| Tema | Estado apos auditoria | Acao |
|---|---|---|
| Identidade academica | Sincronizada no README/CITATION/LICENSE. | Nenhuma adicional no repositorio. |
| Datasets HERIDAL/VisDrone | Sincronizados em `REPRODUCIBILITY.md`. | Nenhuma. |
| Classes VisDrone | Sincronizadas: dez classes preservadas no codigo atual; `pedestrian` e `people` separados no historico. | Nenhuma. |
| Checkpoints e politica de selecao | Sincronizados em tabela historica. | Publicar pesos correspondentes fora do Git e manter hashes. |
| Metricas e thresholds | Sincronizados. | Nenhuma. |
| Ambiente computacional | Sincronizado com ressalva de versoes recuperadas. | Nenhuma. |
| SSD300/VisDrone | Sincronizado como limitacao historica critica. | Reexecutar apenas se desejar novo resultado corrigido. |
| YOLO/AdamW | Divergencia material documentada. | Revisao humana recomendada antes de afirmacao publica adicional. |

## Classificacao das mudancas

- Documentacao: `README.md`, `REPRODUCIBILITY.md`, `KNOWN_ISSUES.md`, `AUDIT_REPORT.md`, `CITATION.cff`, `LICENSE`, `CHANGELOG.md`, `SHA256SUMS.txt`.
- Codigo cientifico ja auditado: remocao de filtros de classe, exposicao de hiperparametros, validacao obrigatoria e preservacao multiclasse.
- Artefatos: nenhum checkpoint, dataset ou metrica historica foi alterado.

## Parecer

O repositorio esta suficientemente sincronizado com a dissertacao para uma publicacao honesta do codigo, desde que a divergencia YOLO/AdamW permaneca explicitamente documentada e que pesos/datasets continuem fora do Git. O ponto que exige revisao humana antes de qualquer afirmacao forte e o otimizador efetivo usado nos checkpoints YOLO historicos.
