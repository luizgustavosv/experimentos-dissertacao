# Resultado executivo

Classificacao final: **SAFE TO PUBLISH AFTER MINOR FIXES**.

Nao foram encontrados segredos por padroes fortes no working tree nem no historico Git varrido localmente. O repositorio ainda nao esta pronto para publicacao imediata por tres motivos praticos: nao ha `LICENSE`, existem dois PRs abertos obsoletos no GitHub mencionando `pedestrian_only`, e ha artefatos locais grandes/ignorados que devem continuar fora do Git. A correcao cientifica sobre filtros de classe e a documentacao de rastreabilidade ja foram aplicadas.

# Problemas encontrados

| Severidade | Categoria | Arquivo/commit | Problema | Risco | Acao necessaria | Resolvido? |
|---|---|---|---|---|---|---|
| HIGH | LICENSE | Repositorio/GitHub | Nao ha arquivo `LICENSE`; GitHub reporta `license: null`. | Codigo publico sem permissao clara de reutilizacao; pode confundir banca e usuarios externos. | Escolher explicitamente MIT, BSD-3-Clause, Apache-2.0, GPL-3.0 ou outra licenca adequada; adicionar `LICENSE` antes da publicacao. | Nao |
| MEDIUM | HISTORY | GitHub PRs #109 e #110 | PRs abertos antigos mencionam `pedestrian_only` e uma correcao que ficou obsoleta pela auditoria atual. | Contradicao publica e ruido cientifico/profissional. | Fechar manualmente os PRs #109 e #110 como obsoletos, referenciando a auditoria/correcao atual. | Nao |
| MEDIUM | REPRODUCIBILITY | `KNOWN_ISSUES.md`; `AUDIT_REPORT.md`; codigo historico | Resultados historicos SSD300/VisDrone nao devem ser interpretados como gerados pelo leitor multiclasse corrigido. | Risco de parecer que resultados foram recalculados ou melhorados apos o experimento. | Manter documentacao de codigo historico vs codigo corrigido; criar tag/release do estado associado a dissertacao. | Parcial |
| MEDIUM | PORTABILITY | Historico Git: `MANUAL_TESTS.md`, `app/detectors/yolo.py`, `app/detectors/train_yolo_visdrone_instrumented.py` | Commits antigos contem exemplos de caminhos locais como unidades Windows e diretorios de dataset. | Baixo risco de privacidade; principalmente portabilidade e estetica historica. | Nao reescrever historico por isso; manter versao atual parametrizada. | Sim no estado atual |
| MEDIUM | HISTORY | Banco de objetos local; arquivo `experimentos-dissertacao.zip` | ZIP local de ~316 MB existe no workspace e aparece como blob no banco de objetos local, mas nao foi associado a commit por `git log --all --full-history`. | Pode aumentar clone/push se algum ref local oculto o carregar; pode conter material indevido se publicado por engano. | Nao versionar; remover do workspace apenas com decisao do autor; executar `git gc/prune` local se confirmado que nao e necessario. | Nao |
| MEDIUM | DOCUMENTATION | `CITATION.cff` | CITATION tinha URL placeholder e autor nao informado. | Apresentacao publica incompleta. | Atualizado para reposititorio real e autor "Luiz Gustavo"; revisar nome academico completo antes da publicacao. | Parcial |
| LOW | PRIVACY | Historico de commits | Autoria alterna entre `luizgsv@protonmail.com` e `108900071+luizgustavosv@users.noreply.github.com`. | Exposicao de e-mail pessoal ja presente no historico. | Aceitar se esse e-mail puder ser publico; reescrever historico so se houver razao forte de privacidade. | Revisar |
| LOW | SECURITY | Codigo Python | Uso amplo de `torch.load` em checkpoints informados pelo usuario. | `torch.load` pode executar payload malicioso se o checkpoint vier de origem nao confiavel; contexto e app local de pesquisa. | Documentar que checkpoints devem ser confiaveis; preferir `weights_only=True` quando compativel. | Parcial |
| LOW | ACTIONS | `.github/workflows` | Nao existe diretorio local `.github`; API de contents retornou 404 para workflows. | Sem risco observado em workflows versionados. Logs remotos antigos nao foram auditados integralmente. | Revisar aba Actions no GitHub antes de tornar publico, se houver runs historicos. | Parcial |
| LOW | DATASET | `app/datasets/`; workspace ignorado | Nao ha evidencia de datasets integrais rastreados; pacote `app/datasets` contem codigo de conversao/readers. | Redistribuicao indevida parece improvavel no Git atual; datasets locais devem ficar fora. | Manter `datasets/` e `data/` ignorados; README deve apontar fontes oficiais, sem redistribuir HERIDAL/VisDrone. | Sim |

# Segredos

Ferramentas externas (`gitleaks`, `trufflehog`, `git-secrets`) nao estavam instaladas no ambiente. Foi executada uma varredura equivalente com `git grep`/`rg` e regexes para GitHub tokens, OpenAI keys, AWS access keys, Google API keys, Hugging Face tokens, JWTs, private keys, URLs com usuario/senha e atribuicoes genericas de `token`, `secret`, `password`, `senha` e `api_key`.

Resultado: **nenhum segredo encontrado** no working tree nem no historico Git local varrido.

Como medida defensiva, se algum segredo tiver sido usado fora deste repositorio durante os experimentos, ele deve ser rotacionado antes da publicacao mesmo sem ter sido encontrado aqui.

# Historico Git

Branches locais:

- `main`: manter.
- `audit-publicacao-cientifica`: manter ate concluir a auditoria e abrir/mesclar PR.

Branches remotos:

- Foram listados cerca de 120 branches `origin/codex/*`.
- Classificacao: **REVIEW MANUALLY / ARCHIVE**. Muitos sao branches antigos de tarefas automatizadas ja mescladas ou obsoletas; nao ha necessidade de reescrever historico por estetica, mas branches abandonados podem confundir leitores quando o repositorio ficar publico.
- Branches #109/#110 associados a PRs abertos devem ser fechados ou marcados como obsoletos antes da publicacao.

Tags:

- Nenhuma tag encontrada.
- Recomenda-se criar uma tag apos a auditoria, por exemplo `dissertation-submission` ou `public-release-1.0`, e documentar se ela representa codigo historico ou codigo corrigido para publicacao.

Commits:

- Mensagens sao majoritariamente tecnicas, algumas informais em portugues, sem achados de segredo.
- Nao recomendo reescrita de historico apenas por caminhos locais ou mensagens imperfeitas.

# Dados pessoais

Serao expostos:

- Usuario GitHub `luizgustavosv`.
- Nome `Luiz Gustavo` em commits/PRs/CITATION.
- E-mail `luizgsv@protonmail.com` em commits recentes.
- E-mail noreply do GitHub em commits anteriores.

Nao foram encontrados CPF, RG, telefone, endereco residencial ou credenciais institucionais por padroes buscados. O e-mail pessoal deve ser aceito conscientemente; alterar autoria historica exigiria reescrita de historico e troca de hashes.

# Licencas

O repositorio ainda nao tem `LICENSE`. Nao escolhi uma licenca automaticamente.

Opcoes comuns:

- MIT: simples e permissiva.
- BSD-3-Clause: permissiva, com clausula contra endosso.
- Apache-2.0: permissiva, com linguagem explicita sobre patentes.
- GPL-3.0: copyleft forte; exige que derivados distribuido preservem a mesma liberdade.

Dependencias principais incluem PyTorch, torchvision, Ultralytics, pycocotools, torchmetrics e outras bibliotecas instaladas via `requirements.txt`. Nao identifiquei codigo vendorizado de terceiros que bloqueie uma licenca permissiva, mas a licenca do proprio repositorio deve ser escolhida pelo autor/orientacao.

# Datasets

Nao ha evidencia de copias integrais de HERIDAL ou VisDrone rastreadas pelo Git atual. O diretorio `app/datasets/` contem codigo de normalizacao/readers/exporters, nao dados originais.

Recomendacao:

- Nao versionar imagens, anotacoes originais completas ou splits redistribuidos sem permissao clara.
- Manter `datasets/` e `data/` no `.gitignore`.
- README deve apontar nome oficial, fonte e estrutura esperada, sem inventar licenca de redistribuicao.

# Reprodutibilidade cientifica

A distincao entre codigo historico e codigo corrigido esta documentada em `README.md`, `KNOWN_ISSUES.md`, `REPRODUCIBILITY.md` e `AUDIT_REPORT.md`. A correcao atual remove filtros de classe e preserva categorias, mas os resultados historicos nao devem ser reinterpretados como se tivessem sido produzidos por essa versao corrigida.

Pendencia: o arquivo da dissertacao nao estava no workspace, portanto a comparacao literal entre texto final e codigo ainda precisa ser feita.

# GitHub Actions

Nao ha `.github/workflows` no estado local atual; consulta a `.github/workflows` via API retornou 404. Portanto, nao foram encontrados workflows versionados com `pull_request_target`, secrets, uploads ou permissoes perigosas.

Limitacao: logs historicos de Actions no GitHub nao foram baixados integralmente. Antes de mudar a visibilidade, revisar manualmente a aba Actions se houver runs antigas.

# Arquivos grandes e artefatos

Artefatos locais ignorados observados:

- `app.log`: ~911 MB. Nao publicar; evidencia local/desenvolvimento.
- `experimentos-dissertacao.zip`: ~316 MB. Nao publicar sem auditoria manual do conteudo.
- `logs/`: multiplos logs e JSONs de predicoes, alguns acima de 100 MB. Manter fora do Git; publicar apenas extratos/metadados necessarios.
- `runs/detect/train4/weights/best.pt` e `last.pt`: ~5,5 MB cada. Se forem pesos cientificamente relevantes, preferir GitHub Release ou LFS, com metadados e hash.

O `.gitignore` foi reforcado para evitar novos commits acidentais de ambientes, logs, checkpoints, datasets, chaves e caches.

# Verificacao final automatizada

Comandos executados ou equivalentes:

- `git branch -a`
- `git tag`
- `git log --oneline --decorate --all`
- `git log --all --format=...`
- `git grep` sobre todos os commits para caminhos locais e padroes sensiveis
- `rg` sobre working tree, incluindo arquivos ignorados fora de `.git`/`.venv`
- varredura de arquivos grandes no workspace
- varredura de objetos historicos por extensoes sensiveis/grandes
- `python -m py_compile` nos modulos alterados
- `pytest -q`

Resultado de testes: `27 passed`.

# Acoes manuais obrigatorias antes de tornar publico

[ ] Escolher e adicionar um arquivo `LICENSE`.
[ ] Fechar ou marcar como obsoletos os PRs abertos #109 e #110.
[ ] Revisar se `luizgsv@protonmail.com` pode ficar publico no historico.
[ ] Confirmar que `experimentos-dissertacao.zip`, `app.log`, `logs/`, `runs/`, `reports/` e datasets locais nao serao adicionados ao commit.
[ ] Revisar a aba Actions/logs no GitHub, se houver runs historicos.
[ ] Revisar Issues/PRs restantes alem da primeira pagina amostrada se houver metadados privados.
[ ] Preencher instituicao/programa e nome academico completo no README/CITATION.
[ ] Comparar a versao final da dissertacao com `KNOWN_ISSUES.md` e `REPRODUCIBILITY.md`.

# Acoes recomendadas, mas nao obrigatorias

[ ] Criar uma tag de rastreabilidade para a versao publica auditada.
[ ] Reduzir/arquivar branches remotos `codex/*` obsoletos.
[ ] Publicar checkpoints relevantes em Release ou Git LFS, com hashes, em vez de versionar no Git comum.
[ ] Adicionar hashes dos datasets preparados e dos checkpoints usados nos experimentos.
[ ] Executar `gitleaks detect --source . --no-git=false` em uma maquina com a ferramenta instalada como segunda opiniao.
[ ] Considerar `torch.load(..., weights_only=True)` onde compativel.

# Parecer final

Eu tornaria este repositorio publico neste momento? **NAO, ainda nao**.

Eu nao vi nenhum indicio de segredo real, chave privada ou dataset indevidamente versionado que bloqueie a publicacao por seguranca imediata. A base tecnica tambem esta em melhor estado apos a auditoria cientifica: filtros `pedestrian_only` foram removidos e a rastreabilidade dos problemas historicos foi documentada.

Mesmo assim, eu aguardaria as pequenas correcoes manuais: adicionar `LICENSE`, fechar PRs obsoletos que contradizem o estado atual e confirmar conscientemente a exposicao do e-mail pessoal no historico. Essas pendencias sao pequenas, mas visiveis para banca e leitores externos.

Depois dessas acoes, meu parecer passa a ser favoravel a publicacao, desde que datasets, logs gigantes, zips e checkpoints locais continuem fora do Git comum ou sejam publicados separadamente com justificativa cientifica.
