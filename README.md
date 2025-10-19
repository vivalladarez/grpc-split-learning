# grpc-split-learning: Multi-Client U-Shaped Split Learning com gRPC, Ray e TensorFlow

Este projeto implementa um sistema completo de Split Learning distribuído no formato U-Shaped, combinando TensorFlow/Keras, gRPC e Ray para permitir o treinamento colaborativo de modelos de deep learning com múltiplos clientes e um servidor central, sem compartilhamento de dados ou rótulos brutos.

---

### 1. Principais recursos

O **Split Learning** divide um modelo neural entre diferentes partes:
- **Clientes:** possuem os dados e os rótulos.
- **Servidor:** mantém apenas as camadas intermediárias.

Neste formato **U-Shaped**, cada cliente:
- Processa os dados locais até um ponto (`M1`),
- Envia as **ativações intermediárias** ao servidor,
- Recebe de volta as ativações processadas (`M2`),
- Calcula a **loss localmente** (pois possui os rótulos),
- E propaga os gradientes de volta para o servidor e seu próprio modelo.
- Todos os clientes reportam métricas ao servidor, que consolida em um único arquivo metrics_all.csv.

### 2. Estrutura principal
    grpc-split-learning/
    ├── splitlearning.proto         # Definição dos serviços e mensagens gRPC
    ├── server.py                   # Servidor (modelo M2, camadas intermediárias e coordenação)
    ├── client.py                   # Cliente (modelos M1 e M3, cálculo local de loss e gradientes)
    ├── driver_ray.py               # Orquestrador com Ray para múltiplos clientes paralelos
    ├── metrics_all.csv             # Métricas globais 
    ├── metrics_client_analysis.py  # Script de análise e visualização das métricas
    ├── plots_results/              # Pasta de saída com gráficos e visualizações geradas automaticamente
    └── README.md                   # Documentação do projeto

### 3. Arquitetura 

                      ─────────────────────────────────────────────────────────────
                                           🖥️  SERVER (server.py)
                      ─────────────────────────────────────────────────────────────
                        • Implementa o modelo intermediário (M2)
                        • Recebe ativações (a₁) dos clientes e calcula z₂ = M2(a₁)
                        • Guarda a₁ por client_id para uso no backward
                        • Recebe gradientes dL/dz₂, realiza backprop em M2
                        • Retorna gradientes dL/da₁ ao cliente
                        • Atualiza pesos de M2
                        • Consolida métricas de todos os clientes em `metrics_all.csv`
                      
                      ─────────────────────────────────────────────────────────────
                                           💻  CLIENTE (client.py)
                      ─────────────────────────────────────────────────────────────
                        • Possui:
                             ├── M1 → parte inicial (convoluções e embedding)
                             └── M3 → parte final (camada de saída + loss)
                        • Executa:
                             1️⃣ Forward local: x → M1(x) → a₁
                             2️⃣ Envia a₁ → Servidor → recebe z₂
                             3️⃣ Calcula loss = CE(y, M3(z₂)) e acurácia
                             4️⃣ Atualiza M3 e envia gradiente dL/dz₂ → Servidor
                             5️⃣ Recebe dL/da₁ e atualiza M1
                        • Mede:
                             - Latência por passo
                             - Bytes enviados (Tx) e recebidos (Rx)
                        • Envia métricas ao servidor via RPC `LogMetrics`
                    
                      ─────────────────────────────────────────────────────────────
                                        ☁️  ORQUESTRADOR (driver_ray.py)
                      ─────────────────────────────────────────────────────────────
                        • Usa **Ray** para executar clientes em paralelo
                        • Cria N atores (`ClientActor`) → cada um representa um cliente
                        • Cada cliente recebe um **shard exclusivo** do CIFAR-10
                        • Coordena o treino distribuído:
                             ├── Inicia todos os clientes simultaneamente
                             ├── Cada cliente treina localmente (Forward + Backward)
                             └── Ray sincroniza o fim de todas as execuções
                        • Evita downloads concorrentes do dataset (driver distribui shards)
                      
                      ─────────────────────────────────────────────────────────────
                                              🔁 FLUXO GERAL
                      ─────────────────────────────────────────────────────────────
                                     DRIVER (Ray)
                                        └── cria → CLIENTE₁ ... CLIENTEₙ
                                                   │
                                                   │  a₁ = M1(x)
                                                   ▼
                                                SERVIDOR (M2)
                                                   │
                                                   │  z₂ = M2(a₁)
                                                   ▼
                                               CLIENTE: loss(y, M3(z₂))
                                                   │
                                                   │  gradiente dL/dz₂
                                                   ▼
                                                SERVIDOR: backprop → dL/da₁
                                                   │
                                                   ▼
                                               CLIENTE: atualiza M1
                      ─────────────────────────────────────────────────────────────
                                  📈 Métricas globais gravadas pelo servidor  
                                                `metrics_all.csv`
                  `timestamp, client_id, epoch, step, loss, acc, latency_s, MB_tx, MB_rx`  
### 4. Setup 

#### Ambiente Virtual e Dependências

Clone o repo e crie e ative o ambiente virtual (Windows):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.bat
```

Instale as dependências:

```bash
# gRPC + Protobuf (evita conflitos com TF)
pip install "protobuf==4.25.3" "grpcio==1.62.2" "grpcio-tools==1.62.2"

# Deep learning
pip install tensorflow==2.17.0 keras==3.3.3

# Orquestração distribuída
pip install ray==2.35.0
```

#### Geração dos Stubs gRPC

O contrato de comunicação está em similarity.proto.
Para gerar as classes Python (splitlearning.py e splitlearning_pb2.py):

```python
python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. splitlearning.proto

```

#### Inicialização e Execução

- Iniciar o Servidor
```bash
python server.py
```

- Executar os Clientes com Ray
```bash
python driver_ray.py --num_clients 3 --epochs 5 --batch_size 64
```
**obs:** aqui é possivel editar o numero de clients, epochs e batch_size

### 5. Resultados

Após a execução do treinamento, o servidor consolida as métricas de todos os clientes no arquivo *metrics_all.csv*. Por fim, oscript metrics_client_analysis.py gera automaticamente os gráficos agregados e salva o resultado em `plots_results/split_learning_metrics.png`

![Resultados](https://github.com/vivalladarez/grpc-split-learning/blob/731fc1101b0a0a4f26b142c4db15e4418ff67aa2/plots_results/split_learning_metrics.png)

Os gráficos acima representam a evolução de Loss, Acurácia, Latência e Bytes Transmitidos ao longo de 4 épocas de treinamento:

- Loss: decresce de forma consistente em todos os clientes, indicando convergência estável mesmo com o modelo dividido entre cliente (M1/M3) e servidor (M2).

- Acurácia: apresenta crescimento contínuo (~0.25 → ~0.55), evidenciando aprendizado global efetivo, mesmo sem troca direta de dados brutos.

- Latência: aumenta levemente após a 2ª época, comportamento esperado devido ao maior sincronismo entre forward e backward federados conforme os pesos se ajustam.

- Transmissão cumulativa (Tx): cresce de maneira quase linear e equilibrada entre os clientes, refletindo a estabilidade da carga de rede do corte M1→M2.
