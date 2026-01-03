import os
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from utils.metrics import evaluate_kshot, evaluate_retrieval, evaluate_clustering


from models.model import SiameseNetwork
from utils.metrics import evaluate_metrics

batch_size = 64
data_dir = "./data/test/"
eval_dir = "./data/eval/"
model_path = "./models/model.pt"
log_dir = "./log/"
os.makedirs(log_dir, exist_ok=True)

@torch.no_grad()
def evaluate(model=None, loader=None):

    all_labels, all_scores = [], []

    for A_x, A_img, B_x, B_img, gate_A_x, gate_A_img, gate_B_x, gate_B_img, label in loader:
        A_x = A_x.cuda()
        A_img = A_img.cuda()
        B_x = B_x.cuda()
        B_img = B_img.cuda()

        gate_A_x = gate_A_x.cuda()
        gate_A_img = gate_A_img.cuda()
        gate_B_x = gate_B_x.cuda()
        gate_B_img = gate_B_img.cuda()

        out = model(
            A_x, gate_A_x,
            A_img, gate_A_img,
            B_x, gate_B_x,
            B_img, gate_B_img
        )

        all_scores.extend(out["similarity"].cpu().numpy())
        all_labels.extend(label.numpy())

    with open(os.path.join(log_dir, "testEval.txt"), "w") as f:
        pr_auc = evaluate_metrics(all_labels, all_scores, f)

    print(f"[Test] PR-AUC: {pr_auc:.4f}")
    return pr_auc


def build_seq(text_data, indices, max_len=5):
    seq = [text_data[i] for i in indices[:max_len]]
    while len(seq) < max_len:
        seq.append(np.zeros_like(seq[0]))
    return np.stack(seq)

def avg_gate(gate_data, indices):
    gates = np.stack([gate_data[i] for i in indices]) 
    return gates.mean(axis=0)                           

@torch.no_grad()
def run_nway_kshot(
    model,
    text_data, img_data,
    text_gate, img_gate,
    labels,
    N, K, Q,
    num_episodes=999,
    top_ks=[1]
):
    episode_results = []

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    support_candidates = {
        l: idxs for l, idxs in label_to_indices.items()
        if len(idxs) >= K
    }

    query_candidates = {
        l: idxs for l, idxs in label_to_indices.items()
        if len(idxs) >= (K + Q)
    }

    support_labels = list(support_candidates.keys())
    query_labels = list(query_candidates.keys())

    assert len(query_labels) >= 1
    assert len(support_labels) >= N

    for _ in range(num_episodes):

        query_label = np.random.choice(query_labels)
        other_support_labels = np.random.choice(
            [l for l in support_labels if l != query_label],
            N - 1,
            replace=False
        )

        episode_labels = [query_label] + list(other_support_labels)

        support_indices = []

        query_support_indices = None

        for l in episode_labels:
            idxs = label_to_indices[l]

            if len(idxs) == K:
                chosen = idxs
            else:
                start = np.random.randint(0, len(idxs) - K + 1)
                chosen = idxs[start:start + K]

            support_indices.append(chosen)

            if l == query_label:
                query_support_indices = set(chosen)

        all_query_idxs = label_to_indices[query_label]
        remaining = [i for i in all_query_idxs if i not in query_support_indices]

        query_indices = []

        if len(remaining) >= Q:
            if len(remaining) == Q:
                query_indices = remaining
            else:
                start = np.random.randint(0, len(remaining) - Q + 1)
                query_indices = remaining[start:start + Q]

        A_x = torch.tensor(
            np.stack([build_seq(text_data, idxs) for idxs in support_indices]),
            dtype=torch.float32
        ).cuda()

        A_img = torch.tensor(
            np.stack([build_seq(img_data, idxs) for idxs in support_indices]),
            dtype=torch.float32
        ).cuda()

        gate_A_x = torch.tensor(
            np.stack([avg_gate(text_gate, idxs) for idxs in support_indices]),
            dtype=torch.float32
        ).cuda()

        gate_A_img = torch.tensor(
            np.stack([avg_gate(img_gate, idxs) for idxs in support_indices]),
            dtype=torch.float32
        ).cuda()

        B_x = torch.tensor(
            np.stack([build_seq(text_data, query_indices)] * N),
            dtype=torch.float32
        ).cuda()

        B_img = torch.tensor(
            np.stack([build_seq(img_data, query_indices)] * N),
            dtype=torch.float32
        ).cuda()

        gate_B_x = torch.tensor(
            np.stack([avg_gate(text_gate, query_indices)] * N),
            dtype=torch.float32
        ).cuda()

        gate_B_img = torch.tensor(
            np.stack([avg_gate(img_gate, query_indices)] * N),
            dtype=torch.float32
        ).cuda()

        out = model(
            A_x, gate_A_x,
            A_img, gate_A_img,
            B_x, gate_B_x,
            B_img, gate_B_img
        )

        scores = out["similarity"].cpu().numpy()

        label_binary = [1 if l == query_label else 0 for l in episode_labels]

        res = evaluate_kshot(scores, label_binary, top_ks)
        episode_results.append(res)

    agg = aggregate_kshot_results(episode_results, top_ks)

    for k in top_ks:
        print(
            f"[{N}-way {K}-shot {Q}-query] "
            f"Acc@{k}: {agg[k]['acc']:.4f} | "
            f"Precision@{k}: {agg[k]['precision']:.4f} | "
            f"Recall@{k}: {agg[k]['recall']:.4f} | "
            f"MAP@{k}: {agg[k]['map']:.4f}"
        )

def aggregate_kshot_results(results_list, top_ks=(1,)):
    agg = {k: {"acc": 0, "precision": 0, "recall": 0, "map": 0} for k in top_ks}
    num = len(results_list)

    for res in results_list:
        for k in top_ks:
            for m in agg[k]:
                agg[k][m] += res[k][m]

    for k in top_ks:
        for m in agg[k]:
            agg[k][m] /= num

    return agg


@torch.no_grad()
def run_full_retrieval(
    model, text_data, img_data,
    text_gate, img_gate, labels,
    top_ks=[1]
):
    num_samples = len(labels)
    all_sim = np.zeros((num_samples, num_samples))

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    for i in range(num_samples):
        A_x = torch.tensor(
            np.repeat(text_data[i:i+1], num_samples, axis=0),
            dtype=torch.float32
        ).cuda()

        A_img = torch.tensor(
            np.repeat(img_data[i:i+1], num_samples, axis=0),
            dtype=torch.float32
        ).cuda()

        gate_A_x = torch.tensor(
            np.repeat(text_gate[i:i+1], num_samples, axis=0),
            dtype=torch.float32
        ).cuda()

        gate_A_img = torch.tensor(
            np.repeat(img_gate[i:i+1], num_samples, axis=0),
            dtype=torch.float32
        ).cuda()

        B_x = torch.tensor(text_data, dtype=torch.float32).cuda()
        B_img = torch.tensor(img_data, dtype=torch.float32).cuda()
        gate_B_x = torch.tensor(text_gate, dtype=torch.float32).cuda()
        gate_B_img = torch.tensor(img_gate, dtype=torch.float32).cuda()

        out = model(
            A_x, gate_A_x,
            A_img, gate_A_img,
            B_x, gate_B_x,
            B_img, gate_B_img
        )

        all_sim[i] = out["similarity"].cpu().numpy()

    results = []

    for i in range(num_samples):
        r = evaluate_retrieval(
            all_sim[i],
            labels,
            i,
            label_to_indices,
            top_ks
        )
        if r is not None:
            results.append(r)

    mrr, mapv, agg = aggregate_retrieval_results(results, top_ks)

    for k in top_ks:
        print(
            f"[retrieval] "
            f"Recall@{k}: {agg[k]['recall']:.4f} | "
            f"Precision@{k}: {agg[k]['precision']:.4f} | "
            f"Accuracy@{k}: {agg[k]['accuracy']:.4f} | "
        )

    print(f"[retrieval] "f"MRR: {mrr:.4f} | MAP: {mapv:.4f}")

def aggregate_retrieval_results(results, top_ks=(1,)):
    valid = len(results)

    mrr = sum(r["mrr"] for r in results) / valid
    mapv = sum(r["map"] for r in results) / valid

    agg = {k: {"recall": 0, "precision": 0, "accuracy": 0} for k in top_ks}
    for r in results:
        for k in top_ks:
            for m in agg[k]:
                agg[k][m] += r["topk"][k][m]

    for k in top_ks:
        for m in agg[k]:
            agg[k][m] /= valid

    return mrr, mapv, agg

@torch.no_grad()
def run_clustering(
    model,
    text_data, img_data,
    text_gate, img_gate,
    labels
):

    model.eval()
    text_data = ensure_sequence(text_data)   # [N, L, D]
    img_data = ensure_sequence(img_data)     # [N, L, D]
    N = len(labels)
    sim_matrix = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        A_x = torch.tensor(
            np.repeat(text_data[i:i+1], N, axis=0),
            dtype=torch.float32
        ).cuda()
        A_img = torch.tensor(
            np.repeat(img_data[i:i+1], N, axis=0),
            dtype=torch.float32
        ).cuda()
        gate_A_x = torch.tensor(
            np.repeat(text_gate[i:i+1], N, axis=0),
            dtype=torch.float32
        ).cuda()
        gate_A_img = torch.tensor(
            np.repeat(img_gate[i:i+1], N, axis=0),
            dtype=torch.float32
        ).cuda()

        B_x = torch.tensor(text_data, dtype=torch.float32).cuda()
        B_img = torch.tensor(img_data, dtype=torch.float32).cuda()
        gate_B_x = torch.tensor(text_gate, dtype=torch.float32).cuda()
        gate_B_img = torch.tensor(img_gate, dtype=torch.float32).cuda()

        out = model(
            A_x, gate_A_x,
            A_img, gate_A_img,
            B_x, gate_B_x,
            B_img, gate_B_img
        )

        sim_matrix[i] = out["similarity"].cpu().numpy()

    num_clusters = len(np.unique(labels))
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity="precomputed",
        random_state=42
    )
    pred_labels = clustering.fit_predict(sim_matrix)

    y_true = LabelEncoder().fit_transform(labels)
    y_pred = LabelEncoder().fit_transform(pred_labels)

    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    acc = evaluate_clustering(y_true, y_pred)

    print(f"[Clustering] ARI={ari:.4f} | AMI={ami:.4f} | ACC={acc:.4f}")
    return ami, ari, acc

def ensure_sequence(x):
    if x.ndim == 3:
        return x[:, None, :, :]
    elif x.ndim == 4:
        return x

if __name__ == "__main__":
    evaluate()
