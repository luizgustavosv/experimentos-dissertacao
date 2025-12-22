# Laboratório de experimentos — Detecção de pessoas em imagens aéreas

Aplicação gráfica em Python para conduzir experimentos controlados com os
arquiteturas YOLO, SSD, Faster R-CNN e RetinaNet. Ela facilita:

- Treinamento com datasets personalizados e salvamento dos pesos.
- Inferência em lotes de imagens com geração de relatório em PDF.
- Validação com gráficos de desempenho (precisão, recall, mAP) e relatório em PDF.
- Normalização de datasets para fluxos de experimento reproduzíveis.

A implementação usa detectores _stub_ para permitir navegação pela interface e
registro de artefatos sem depender de downloads pesados. Substitua as chamadas
pelas implementações reais (por exemplo, `ultralytics.YOLO`, `torchvision.models`
SSD/Faster R-CNN/RetinaNet) quando quiser treinar de fato.

## Como executar

1. Crie um ambiente virtual e instale as dependências:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Inicie a interface gráfica:

   ```bash
   python -m app.main
   ```

3. Escolha o algoritmo, a ação (Treinar, Inferir, Validar ou Normalizar) e
   preencha os caminhos solicitados. A aplicação gera relatórios PDF e gráficos
   simples para registrar os experimentos.

### Estrutura de código

- `app/gui.py`: interface Tkinter e roteamento das ações.
- `app/controller.py`: orquestra os detectores e resultados.
- `app/detectors/`: declaracão dos algoritmos e _stubs_.
- `app/reporting/reports.py`: geração de gráficos e relatórios PDF.
- `app/metrics.py`: estrutura de métricas usadas na aplicação.

### Observações sobre modelos reais

Os detectores reais podem ser obtidos em repositórios abertos:

- YOLOv12n: `https://github.com/ultralytics/ultralytics`
- SSD / Faster R-CNN / RetinaNet: modelos disponíveis em `torchvision` e
  tutoriais em `https://github.com/pytorch/vision/tree/main/references/detection`

Substitua os _stubs_ no diretório `app/detectors` quando adicionar os modelos
reais e ajuste as chamadas de treino/validação conforme o seu pipeline.
