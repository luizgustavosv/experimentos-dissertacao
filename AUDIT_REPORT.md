# Audit Report

Contexto: auditoria para publicacao do codigo experimental da dissertacao. Branch: `audit-publicacao-cientifica`. Commit base observado: `20b18aae0665f62374b3e1c427b9359fcb5c5b64`.

Limitacao importante: o arquivo da dissertacao nao estava disponivel no workspace. A comparacao texto-versus-codigo ainda deve ser executada quando o documento for fornecido.

| Severidade | Arquivo/linha | Problema | Impacto cientifico potencial | Experimentos afetados | Correcao aplicada | Altera resultados historicos? |
|---|---|---|---|---|---|---|
| CRITICAL | `app/datasets/readers/heridal_reader.py:25`; `normalize_dataset.py:223`; `app/detectors/ssd.py:156` | Fluxos de normalizacao/leitura podiam colapsar ou restringir classes para `human`, incompativel com avaliacao multiclasse. | Perda silenciosa de anotacoes/categorias e metricas por classe invalidas. | HERIDAL/VisDrone quando usado em fluxos normalizados. | Readers/normalizadores atuais preservam classes; testes atualizados. | Sim, se aplicado retroativamente. Deve ser tratado como correcao posterior. |
| CRITICAL | Defeito historico reportado para SSD300/VisDrone | Treinamento historico reteve apenas `pedestrian`. | Resultado SSD300/VisDrone nao representa detector multiclasse completo. | SSD300/VisDrone historico. | Documentado em `KNOWN_ISSUES.md`; teste multiclasse criado. | Nao altera o historico; impede reinterpretacao indevida. |
| HIGH | `app/gui.py:609`; `app/controller.py:29`; `app/detectors/yolo.py:170` | Hiperparametros reais de treino nao estavam todos selecionaveis/propagados pela GUI. | Dificulta reproducao e pode ocultar diferencas entre execucoes. | Todos os modelos. | Todos os campos de `TrainConfig` foram expostos; overrides propagados ao controller/backends. | Pode alterar novas execucoes; historico deve usar configs registradas. |
| HIGH | `app/controller.py:78`, `app/controller.py:263`, `app/controller.py:309` | Validacao COCO podia usar resolucao automatica de anotacoes. | Ambiguidade de split de validacao e risco de avaliar conjunto errado. | Faster R-CNN/RetinaNet treino e validacao pos-treino. | Anotacoes de validacao agora sao obrigatorias. | Pode alterar novas execucoes; nao reclassifica historico. |
| HIGH | `app/detectors/utils.py:258`; `app/gui.py` | Opcao/filtro `pedestrian_only` permitia restringir predicoes por classe. | Comparacoes poderiam misturar avaliacao single-class e multiclasse. | Inferencia/benchmark YOLO e Torchvision; SSD especialmente. | Opcao removida e filtro por classe removido da funcao de predicao. | Pode alterar novas inferencias; historico precisa declarar se usou filtro. |
| MEDIUM | `app/detectors/torchvision_train.py:2808`; `app/detectors/config.py:13` | Momentum do SGD era constante magica e nao configuravel. | Reproducibilidade incompleta do treino Torchvision. | SSD300, Faster R-CNN, RetinaNet. | `momentum` adicionado ao `TrainConfig` e GUI. | Novas execucoes podem mudar se usuario alterar o valor. |
| MEDIUM | `tools/analisar_ancoras_caixas.py`; `consolidar_historicos_treinamento.py` | Scripts auxiliares dependiam de caminhos absolutos locais. | Dificultava reproducao por terceiros; podia expor organizacao local. | Auditorias e consolidacoes auxiliares. | Parametrizado por argumentos CLI ou variaveis de ambiente; `.gitignore` reforcado. | Nao. |
| MEDIUM | `.gitignore` | Artefatos grandes/locais nao estavam todos explicitamente ignorados. | Risco de publicar zips, reports, checkpoints ou dados locais. | Publicacao do repositorio. | Adicionados `*.zip`, `*.pt`, `*.pth`, `reports/`, `datasets/`, `data/`, `.pytest_cache/`. | Nao. |
| LOW | `README.md`, `REPRODUCIBILITY.md`, `KNOWN_ISSUES.md`, `CITATION.cff` | Documentacao publica insuficiente/desatualizada. | Banca e terceiros poderiam interpretar o repositorio como mais reprodutivel do que era. | Publicacao. | Documentacao criada/revisada com limitacoes. | Nao. |

## Impacto sobre a dissertacao

- Problemas ja reconhecidos na dissertacao: segundo o pedido, o defeito SSD300/VisDrone que reteve apenas `pedestrian` ja esta registrado. O repositorio agora documenta isso explicitamente.
- Problemas que podem exigir alteracao do texto: qualquer trecho que descreva a versao publica atual como identica ao codigo historico deve distinguir "codigo usado nos experimentos" de "codigo corrigido para publicacao". A confirmacao exata depende do arquivo da dissertacao.
- Problemas apenas da versao publica: README antigo mencionava stubs e nao descrevia adequadamente os pipelines atuais; `.gitignore` nao isolava todos os artefatos recomendados.
- Numeros que podem precisar recalculo: novas execucoes multiclasse de SSD300/VisDrone devem ser recalculadas se forem apresentadas como resultado corrigido. Os numeros historicos nao devem ser substituidos silenciosamente.
- Resultado potencialmente nao defensavel: SSD300/VisDrone como comparacao multiclasse completa nao e defensavel se derivado do leitor historico `pedestrian`-only. Ele pode ser defensavel apenas como resultado historico com limitacao explicitada.

## Classificacao das mudancas desta auditoria

- A: README, REPRODUCIBILITY, KNOWN_ISSUES, CITATION, .gitignore.
- B: testes de regressao, py_compile, busca de residuos, GUI rolavel.
- C: remocao de filtros por classe, preservacao multiclasse, obrigatoriedade de anotacoes de validacao, exposicao/propagacao de hiperparametros.
- D: documentacao e protecao contra regressao do defeito historico SSD300/VisDrone.
