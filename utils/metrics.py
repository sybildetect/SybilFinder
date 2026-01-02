import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment

def evaluate_metrics(y_true, sim_scores, log_writer=None, threshold=0.9):
    y_true = np.array(y_true)
    sim_scores = np.array(sim_scores)
    preds = (sim_scores >= threshold).astype(np.float32)

    roc_auc = roc_auc_score(y_true, sim_scores)
    pr_auc = average_precision_score(y_true, sim_scores)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    msg = (f"[Eval] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
           f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(msg)

    if log_writer:
        log_writer.write(msg + "\n")
        log_writer.flush()

    return pr_auc

def evaluate_kshot(
    scores,
    label_binary,
    top_ks=(1,)
):

    scores = np.asarray(scores)
    label_binary = np.asarray(label_binary)

    sorted_idx = np.argsort(scores)[::-1]
    sorted_labels = label_binary[sorted_idx]

    results = {}
    for k in top_ks:
        topk = sorted_labels[:k]
        results[k] = {
            "acc": float(1 in topk),
            "precision": float(np.mean(topk)),
            "recall": float(np.sum(topk) / np.sum(label_binary)),
            "map": float(average_precision_score(label_binary, scores)),
        }
    return results

def evaluate_retrieval(
    sims,
    labels,
    query_index,
    label_to_indices,
    top_ks=(1,)
):
    sims = sims.copy()
    sims[query_index] = -1e9

    sorted_idx = np.argsort(sims)[::-1]
    true_label = labels[query_index]
    relevant = set(label_to_indices[true_label]) - {query_index}

    if len(relevant) == 0:
        return None

    res = {
        "mrr": 0.0,
        "map": average_precision_score(
            [1 if labels[j] == true_label else 0 for j in range(len(labels))],
            sims
        ),
        "topk": {}
    }

    for rank, idx in enumerate(sorted_idx):
        if labels[idx] == true_label:
            res["mrr"] = 1.0 / (rank + 1)
            break

    for k in top_ks:
        topk = sorted_idx[:k]
        hit = sum(1 for i in topk if labels[i] == true_label)
        res["topk"][k] = {
            "recall": hit / len(relevant),
            "precision": hit / k,
            "accuracy": float(hit > 0),
        }

    return res

def evaluate_clustering(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size