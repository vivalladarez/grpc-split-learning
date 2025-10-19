# driver_ray.py (versão simples)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import ray
from tensorflow import keras

from client import SplitClient

@ray.remote(num_gpus=0)
class ClientActor:
    def __init__(self, client_id: int, server_addr: str, batch_size: int, lr: float,
                 x_shard, y_shard):
        # Recebe arrays diretamente (NÃO usa ray.get aqui)
        self.x = x_shard
        self.y = y_shard
        self.client_id = client_id
        self.batch_size = batch_size
        self.cli = SplitClient(client_id=client_id, server_addr=server_addr,
                               batch_size=batch_size, lr=lr)

    def train(self, epochs: int):
        bs = self.batch_size
        steps = max(1, len(self.x) // bs)
        for epoch in range(epochs):
            idx = np.random.permutation(len(self.x))
            self.x, self.y = self.x[idx], self.y[idx]
            for s in range(steps):
                xb = self.x[s*bs:(s+1)*bs]
                yb = self.y[s*bs:(s+1)*bs]
                loss, acc = self.cli.train_step(xb, yb, epoch=epoch, step=s)
                if s % 20 == 0:
                    print(f"[Client {self.client_id}] epoch={epoch} step={s}/{steps} loss={loss:.4f} acc={acc:.4f}")
        return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="localhost:50051")
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Inicia o Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Carrega o dataset uma única vez no driver
    (x_all, y_all), _ = keras.datasets.cifar10.load_data()
    x_all = x_all.astype("float32") / 255.0
    y_all = y_all.astype("int32").squeeze(-1)

    # Sharding simples e passagem direta para os atores (sem ray.put)
    n = len(x_all)
    per = n // args.num_clients
    actors = []
    for i in range(args.num_clients):
        start = per * i
        end = per * (i+1) if i < args.num_clients - 1 else n
        xs, ys = x_all[start:end], y_all[start:end]
        actors.append(
            ClientActor.remote(
                client_id=i+1,
                server_addr=args.server,
                batch_size=args.batch_size,
                lr=args.lr,
                x_shard=xs,
                y_shard=ys,
            )
        )

    # Treina em paralelo
    ray.get([a.train.remote(args.epochs) for a in actors])

    ray.shutdown()
