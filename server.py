# server.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import csv
import time
import grpc
from concurrent import futures
import threading
import numpy as np
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

# --------- Modelo do Servidor: M2 -------------------------
def build_M2(input_dim=128, hidden=128):
    inp = keras.Input(shape=(input_dim,), name="m2_in")
    x   = keras.layers.Dense(hidden, activation="relu")(inp)
    out = x  # saída z2 (dimensionamento mantido)
    return keras.Model(inp, out, name="M2_server")

class SplitLearningService(pb2_grpc.SplitLearningServicer):
    def __init__(self, input_dim=128):
        # Modelo/otimizador
        self.model = build_M2(input_dim=input_dim, hidden=input_dim)
        self.optimizer = keras.optimizers.Adam(1e-3)

        # Locks e caches
        self._lock = threading.Lock()        # protege atualizações e escrita de CSV
        self._a1_cache = {}                  # cache a1 por client_id

        # Caminho do CSV consolidado
        self.metrics_path = os.path.join(os.path.dirname(__file__), "metrics_all.csv")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["timestamp","client_id","epoch","step","loss","acc","latency_s","MB_tx","MB_rx"])

    # ---------- Forward: recebe a1, devolve z2 e guarda a1 em cache ----------
    def Forward(self, request, context):
        client_id = request.client_id
        a1 = pb_to_tensor(request.a1)

        with self._lock:
            z2 = self.model(a1, training=True)
            # guarda a1 deste minibatch para este cliente (apenas o mais recente)
            self._a1_cache[client_id] = a1

        resp = pb2.ForwardResponse()
        resp.z2.CopyFrom(tensor_to_pb(z2))
        return resp

    # ---------- Backward: recebe dz2, atualiza M2 e retorna da1 ----------
    def Backward(self, request, context):
        client_id = request.client_id
        dz2 = pb_to_tensor(request.dz2)

        with self._lock:
            if client_id not in self._a1_cache:
                context.set_details(f"a1 cache not found for client_id={client_id}")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                return pb2.BackwardResponse()

            a1 = self._a1_cache.pop(client_id)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(a1)
                z2 = self.model(a1, training=True)
                # loss proxy para propagar dz2 através de M2 e até a1
                loss_proxy = tf.reduce_sum(z2 * tf.stop_gradient(dz2))

            grads_m2 = tape.gradient(loss_proxy, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads_m2, self.model.trainable_variables))
            da1 = tape.gradient(loss_proxy, a1)

        resp = pb2.BackwardResponse()
        resp.da1.CopyFrom(tensor_to_pb(da1))
        return resp

    # ---------- LogMetrics - recebe e grava métricas de todos os clientes ----------
    def LogMetrics(self, request, context):
        row = [
            float(request.timestamp),
            int(request.client_id),
            int(request.epoch),
            int(request.step),
            float(request.loss),
            float(request.acc),
            float(request.latency_s),
            float(request.mb_tx),
            float(request.mb_rx),
        ]
        with self._lock:
            with open(self.metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        return pb2.Ack()

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=16),
        options=[
            ('grpc.max_send_message_length', 20 * 1024 * 1024),
            ('grpc.max_receive_message_length', 20 * 1024 * 1024),
        ],
    )
    pb2_grpc.add_SplitLearningServicer_to_server(SplitLearningService(input_dim=128), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("✅ Split-Learning gRPC Server ON 0.0.0.0:50051")
    print(f"📈 Métricas consolidadas em: {os.path.abspath(os.path.join(os.path.dirname(__file__), 'metrics_all.csv'))}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
