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
    ├── splitlearning.proto        # Definição dos serviços e mensagens gRPC
    ├── server.py                  # Servidor (modelo M2 + logging centralizado)
    ├── client.py                  # Cliente (modelos M1/M3 + envio de métricas)
    ├── driver_ray.py              # Orquestrador Ray (múltiplos clientes)
    ├── metrics_all.csv            # Métricas consolidadas (gerado em runtime)
    └── README.md
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
python driver_ray.py --num_clients 4 --epochs 2 --batch_size 64
```
