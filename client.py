# client.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import grpc
import tensorflow as tf
from tensorflow import keras

import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc

# --------- Utilidades (serialização de tensores) ---------
def tensor_to_pb(t: tf.Tensor) -> pb2.Tensor:
    arr = t.numpy().astype(np.float32, copy=False)
    msg = pb2.Tensor()
    msg.data.extend(arr.flatten().tolist())
    msg.shape.extend(list(arr.shape))
    return msg

def pb_to_tensor(msg: pb2.Tensor) -> tf.Tensor:
    shape = tuple(msg.shape)
    arr = np.asarray(msg.data, dtype=np.float32).reshape(shape)
    return tf.convert_to_tensor(arr, dtype=tf.float32)

# --------- Split simples de dados entre clientes (IID) ---------
def get_shard(x, y, num_clients, client_id):
    """Divide (x,y) em fatias quase iguais. client_id começa em 1."""
    n = len(x)
    per = n // num_clients
    start = per * (client_id - 1)
    end = per * client_id if client_id < num_clients else n
    return x[start:end], y[start:end]

# --------- Modelos do Cliente: M1 (head) e M3 (tail) ---------
def build_M1(input_shape=(32, 32, 3), embed_dim=128):
    inp = keras.Input(shape=input_shape, name="m1_in")
    x   = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x   = keras.layers.MaxPooling2D((2, 2))(x)
    x   = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x   = keras.layers.MaxPooling2D((2, 2))(x)
    x   = keras.layers.Flatten()(x)
    x   = keras.layers.Dense(embed_dim, activation="relu")(x)
    return keras.Model(inp, x, name="M1_client")

def build_M3(input_dim=128, num_classes=10):
    inp = keras.Input(shape=(input_dim,), name="m3_in")
    out = keras.layers.Dense(num_classes, activation="softmax")(inp)
    return keras.Model(inp, out, name="M3_client")

class SplitClient:
    def __init__(self, client_id: int, server_addr: str = "localhost:50051",
                 batch_size: int = 64, lr: float = 1e-3):
        self.client_id = client_id
        self.batch_size = batch_size

        # gRPC stub
        self.channel = grpc.insecure_channel(
            server_addr,
            options=[
                ('grpc.max_send_message_length', 20 * 1024 * 1024),
                ('grpc.max_receive_message_length', 20 * 1024 * 1024),
            ]
        )
        self.stub = pb2_grpc.SplitLearningStub(self.channel)

        # modelos locais e otimizadores
        self.M1 = build_M1(embed_dim=128)
        self.M3 = build_M3(input_dim=128, num_classes=10)
        self.opt_M1 = keras.optimizers.Adam(lr)
        self.opt_M3 = keras.optimizers.Adam(lr)

        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy()

    def train_step(self, x, y, epoch: int = 0, step: int = 0):
        t_start = time.time()

        # 1) Forward local M1 -> a1 -> envia a1 -> recebe z2
        with tf.GradientTape(persistent=True) as tape1:
            a1 = self.M1(x, training=True)
        fwd_resp = self.stub.Forward(pb2.ForwardRequest(
            client_id=self.client_id,
            a1=tensor_to_pb(a1)
        ))
        z2 = pb_to_tensor(fwd_resp.z2)

        # 2) Loss local + atualização de M3; gera dz2
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(z2)
            logits = self.M3(z2, training=True)
            loss = self.loss_fn(y, logits)
        grads_M3 = tape2.gradient(loss, self.M3.trainable_variables)
        self.opt_M3.apply_gradients(zip(grads_M3, self.M3.trainable_variables))
        dz2 = tape2.gradient(loss, z2)

        # 3) Backward remoto (server atualiza M2) -> recebe da1
        bwd_resp = self.stub.Backward(pb2.BackwardRequest(
            client_id=self.client_id,
            dz2=tensor_to_pb(dz2)
        ))
        da1 = pb_to_tensor(bwd_resp.da1)

        # 4) Atualiza M1 usando da1
        grads_M1 = tape1.gradient(a1, self.M1.trainable_variables, output_gradients=da1)
        self.opt_M1.apply_gradients(zip(grads_M1, self.M1.trainable_variables))

        # 5) Métricas + envio centralizado ao servidor
        self.acc_metric.update_state(y, logits)
        acc_val = float(self.acc_metric.result().numpy())
        loss_val = float(loss.numpy())
        latency = time.time() - t_start
        mb_tx = a1.numpy().nbytes / 2**20
        mb_rx = da1.numpy().nbytes / 2**20
        ts = time.time()

        # Envia métricas via RPC (gerará um único CSV no servidor)
        try:
            self.stub.LogMetrics(pb2.MetricsLog(
                client_id=self.client_id,
                epoch=epoch,
                step=step,
                loss=loss_val,
                acc=acc_val,
                latency_s=latency,
                mb_tx=mb_tx,
                mb_rx=mb_rx,
                timestamp=ts,
            ))
        except Exception as e:
            # fail-safe: não interrompe o treino se o log falhar
            print(f"[WARN][client {self.client_id}] falha ao enviar métricas: {e}")

        return loss_val, acc_val

if __name__ == "__main__":
    # pequeno teste local (opcional)
    (x, y), _ = keras.datasets.cifar10.load_data()
    x = x.astype("float32") / 255.0
    y = y.astype("int32").squeeze(-1)
    x, y = x[:128], y[:128]

    cli = SplitClient(client_id=1)
    bs = 64
    xb, yb = x[:bs], y[:bs]
    loss, acc = cli.train_step(xb, yb, epoch=0, step=0)
    print(f"[sanity-check] loss={loss:.4f} acc={acc:.4f}")
