import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(style="whitegrid")
plt.rcParams.update({"font.size": 12})

CSV_PATH = "metrics_all.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("Coloque o arquivo metrics_all.csv na mesma pasta deste notebook/script.")

df = pd.read_csv(CSV_PATH)
expected = ["timestamp", "client_id", "epoch", "step", "loss", "acc", "latency_s", "MB_tx", "MB_rx"]
df.columns = expected[:len(df.columns)]

df["client_id"]  = df["client_id"].astype(int)
df["epoch"]      = df["epoch"].astype(int)
df["step"]       = df["step"].astype(int)
for c in ["loss", "acc", "latency_s", "MB_tx", "MB_rx", "timestamp"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["MB_total"] = df["MB_tx"] + df["MB_rx"]

agg = (
    df.groupby(["client_id", "epoch"])
      .agg({"loss": "mean", "acc": "mean", "latency_s": "mean", "MB_tx": "sum"})
      .reset_index()
      .sort_values(["client_id", "epoch"])
)

client_ids = sorted(agg["client_id"].unique())
base_palette = sns.color_palette("tab10", n_colors=max(10, len(client_ids)))
palette_map = {cid: base_palette[i % len(base_palette)] for i, cid in enumerate(client_ids)}

def plot_with_fixed_colors(ax, data, x, y, ylabel):
    for cid in client_ids:
        g = data[data["client_id"] == cid]
        if not g.empty:
            ax.plot(g[x], g[y], linewidth=2,
                    label=f"Cliente {cid}", color=palette_map[cid])
    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":")
    ax.legend(title="Cliente")

fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax = ax.flatten()

# Loss
plot_with_fixed_colors(ax[0], agg, x="epoch", y="loss", ylabel="Loss")

# Acurácia
plot_with_fixed_colors(ax[1], agg, x="epoch", y="acc", ylabel="Acurácia (%)")

# Latência
plot_with_fixed_colors(ax[2], agg, x="epoch", y="latency_s", ylabel="Latência (s)")

# Tx cumulativo (tempo relativo)
per_client_time = df.sort_values(["client_id", "timestamp"]).copy()
t0 = per_client_time["timestamp"].min()
per_client_time["t_rel"] = per_client_time["timestamp"] - t0
per_client_time["MB_tx_cum"] = per_client_time.groupby("client_id")["MB_tx"].cumsum()

for cid in client_ids:
    g = per_client_time[per_client_time["client_id"] == cid]
    if not g.empty:
        ax[3].plot(g["t_rel"], g["MB_tx_cum"], linewidth=2,
                   label=f"Cliente {cid}", color=palette_map[cid])

ax[3].set_xlabel("Tempo (s) — relativo")
ax[3].set_ylabel("MB (Tx) — cumulativo")
ax[3].set_yscale("log")
ax[3].grid(True, linestyle=":")
ax[3].legend(title="Cliente")

plt.suptitle("Métricas — Split Learning (por cliente)", fontsize=15, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])


save_dir = os.path.join("plots_results")

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "split_learning_metrics.png")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Gráfico salvo em: {save_path}")

plt.show()
