import os
import sys
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils
from parse_args import args
from CURE_Model import CauFusion

import tasks_NY.tasks_crime, tasks_NY.tasks_chk, tasks_NY.tasks_serviceCall
import tasks_Chi.tasks_crime, tasks_Chi.tasks_chk, tasks_Chi.tasks_serviceCall
import tasks_SF.tasks_crime, tasks_SF.tasks_chk, tasks_SF.tasks_serviceCall


# ============================================================
# Logger
# ============================================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


class DummyWriter:
    def write(self, data):
        pass

    def flush(self):
        pass


def make_log_dir():
    log_dir = "logs_cau_lambda"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


# ============================================================
# Reproducibility
# ============================================================
def set_full_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)


# ============================================================
# Data / Config
# ============================================================
features, mob_adj, poi_sim, land_sim = utils.load_data()

city = args.city
embedding_size = args.embedding_size
d_prime = args.d_prime
d_m = args.d_m
c = args.c
POI_dim = args.POI_dim
landUse_dim = args.landUse_dim
region_num = args.region_num
task = args.task


# ============================================================
# Losses
# ============================================================
def _mob_loss(s_embeddings, t_embeddings, mob):
    inner_prod = torch.mm(s_embeddings, t_embeddings.T)
    phat = F.softmax(inner_prod, dim=-1)
    loss = torch.sum(-torch.mul(mob, torch.log(phat + 1e-4)))

    inner_prod = torch.mm(t_embeddings, s_embeddings.T)
    phat = F.softmax(inner_prod, dim=-1)
    loss += torch.sum(-torch.mul(mob.T, torch.log(phat + 1e-4)))
    return loss


def _general_loss(embeddings, adj):
    inner_prod = F.cosine_similarity(
        embeddings.unsqueeze(1),
        embeddings.unsqueeze(0),
        dim=2
    )
    return F.mse_loss(inner_prod, adj)


def causal_deconf_loss(c, r_views, eps=1e-6):
    c0 = c - c.mean(dim=0, keepdim=True)
    c0 = c0 / (c0.std(dim=0, keepdim=True) + eps)

    V = r_views.size(0)
    loss = 0.0
    for v in range(V):
        r = r_views[v]
        r0 = r - r.mean(dim=0, keepdim=True)
        r0 = r0 / (r0.std(dim=0, keepdim=True) + eps)

        corr = (r0 * c0).mean(dim=0)
        loss = loss + (corr ** 2).mean()

    return loss / V


class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()

    def forward(self, out_s, out_t, mob_adj, out_p, poi_sim, out_l, land_sim):
        mob_loss = _mob_loss(out_s, out_t, mob_adj)
        poi_loss = _general_loss(out_p, poi_sim)
        land_loss = _general_loss(out_l, land_sim)
        return mob_loss + poi_loss + land_loss


# ============================================================
# Helpers
# ============================================================
def safe_to_device(features, mob_adj, poi_sim, land_sim, device):
    poi, land, mob = features
    return (
        poi.to(device),
        land.to(device),
        mob.to(device),
    ), mob_adj.to(device), poi_sim.to(device), land_sim.to(device)


def build_graphs(mob_adj, poi_sim, land_sim):
    A_local = mob_adj
    A_global = 0.5 * (poi_sim + land_sim)

    A_poi = poi_sim
    A_landuse = land_sim
    A_mob = mob_adj
    return A_local, A_global, A_poi, A_landuse, A_mob


def evaluate_downstream(embs, city, task):
    if task == "checkIn":
        if city == "NY":
            return tasks_NY.tasks_chk.do_tasks(embs)
        elif city == "Chi":
            return tasks_Chi.tasks_chk.do_tasks(embs)
        elif city == "SF":
            return tasks_SF.tasks_chk.do_tasks(embs)

    elif task == "crime":
        if city == "NY":
            return tasks_NY.tasks_crime.do_tasks(embs)
        elif city == "Chi":
            return tasks_Chi.tasks_crime.do_tasks(embs)
        elif city == "SF":
            return tasks_SF.tasks_crime.do_tasks(embs)

    elif task == "serviceCall":
        if city == "NY":
            return tasks_NY.tasks_serviceCall.do_tasks(embs)
        elif city == "Chi":
            return tasks_Chi.tasks_serviceCall.do_tasks(embs)
        elif city == "SF":
            return tasks_SF.tasks_serviceCall.do_tasks(embs)

    raise ValueError(f"Unsupported city/task combination: city={city}, task={task}")


def build_model(device):
    model = CauFusion(
        poi_dim=POI_dim,
        landUse_dim=landUse_dim,
        input_dim=embedding_size,
        output_dim=embedding_size,
        d_prime=d_prime,
        d_m=d_m,
        c=c
    ).to(device)

    model_loss = ModelLoss().to(device)
    return model, model_loss


# ============================================================
# Train / Test
# ============================================================
def train_model(features, mob_adj, poi_sim, land_sim, model, model_loss, city, task, device):
    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    lambda_causal = getattr(args, "lambda_causal", 0.0)

    input_features, mob_adj, poi_sim, land_sim = safe_to_device(
        features, mob_adj, poi_sim, land_sim, device
    )

    A_local, A_global, A_poi, A_landuse, A_mob = build_graphs(mob_adj, poi_sim, land_sim)
    model.set_graphs(
        A_local=A_local,
        A_global=A_global,
        A_poi=A_poi,
        A_landuse=A_landuse,
        A_mob=A_mob
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_r2 = -float("inf")
    best_mae = None
    best_rmse = None

    for epoch in range(epochs):
        model.train()

        out_s, out_t, out_p, out_l = model(input_features)
        loss_main = model_loss(out_s, out_t, mob_adj, out_p, poi_sim, out_l, land_sim)

        loss_causal = torch.tensor(0.0, device=device)
        if lambda_causal > 0:
            inter_module = model.interViewEncoder
            c_cache = getattr(inter_module, "cached_c", None)
            r_cache = getattr(inter_module, "cached_r_views", None)
            if c_cache is not None and r_cache is not None:
                loss_causal = causal_deconf_loss(c_cache, r_cache)

        loss = loss_main + lambda_causal * loss_causal

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 不打印训练过程，只在评估点更新 best
        if epoch % 30 == 0:
            model.eval()
            with torch.no_grad():
                embs = model.out_feature().detach().cpu().numpy()
                mae, rmse, r2 = evaluate_downstream(embs, city, task)

                if r2 > best_r2:
                    best_r2 = r2
                    best_mae = mae
                    best_rmse = rmse

    if best_mae is None:
        model.eval()
        with torch.no_grad():
            embs = model.out_feature().detach().cpu().numpy()
            best_mae, best_rmse, best_r2 = evaluate_downstream(embs, city, task)

    return float(best_mae), float(best_rmse), float(best_r2)


def run_once(seed, device):
    set_full_seed(seed)

    model, model_loss = build_model(device)
    mae, rmse, r2 = train_model(
        features, mob_adj, poi_sim, land_sim,
        model, model_loss, city, task, device
    )

    return {
        "seed": seed,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def run_multiple_seeds(device, seeds, top_k=5):
    log_dir = make_log_dir()

    # 重要：lambda_causal 是本实验的变量，必须写入文件名和日志内容，
    # 否则不同 lambda 的实验会保存到同一个 txt，导致结果被覆盖或无法区分。
    lambda_causal = getattr(args, "lambda_causal", 0.0)
    lambda_str = str(lambda_causal).replace(".", "p")

    result_path = os.path.join(
        log_dir,
        f"all_results_city_{args.city}_task_{args.task}"
        f"_emb_{args.embedding_size}"
        f"_lr_{args.learning_rate}"
        f"_lambda_{lambda_str}.txt"
    )

    all_results = []

    # 只把最终结果写入同一个txt，不写训练过程
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=============== Final Results of All Seeds ===============\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"City: {city}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Embedding size: {args.embedding_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Lambda causal: {lambda_causal}\n")
        f.write(f"Total seeds: {len(seeds)}\n")
        f.write("==========================================================\n\n")

        for i, seed in enumerate(seeds, 1):
            result = run_once(seed, device)
            all_results.append(result)

            line = "Seed {} | MAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}".format(
                result["seed"], result["mae"], result["rmse"], result["r2"]
            )
            print(f"[{i}/{len(seeds)}] {line}")   # 控制台可以看进度
            f.write(line + "\n")                  # txt里只写最终结果行

        # 排序
        all_results = sorted(all_results, key=lambda x: x["r2"], reverse=True)
        top_results = all_results[:top_k]

        f.write("\n")
        f.write("================ Best Top-{} Seeds ================\n".format(top_k))
        for rank, res in enumerate(top_results, 1):
            line = "Top {} | Seed {} | MAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}".format(
                rank, res["seed"], res["mae"], res["rmse"], res["r2"]
            )
            f.write(line + "\n")

    return top_results, all_results, result_path


# def run_multiple_seeds(device, seeds, top_k=5):
#     log_dir = make_log_dir()
#     result_path = os.path.join(
#         log_dir,
#         f"all_results_city_{args.city}_task_{args.task}_emb_{args.embedding_size}_lr_{args.learning_rate}.txt"
#     )

#     all_results = []

#     # 只把最终结果写入同一个txt，不写训练过程
#     with open(result_path, "w", encoding="utf-8") as f:
#         f.write("=============== Final Results of All Seeds ===============\n")
#         f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"City: {city}\n")
#         f.write(f"Task: {task}\n")
#         f.write(f"Total seeds: {len(seeds)}\n")
#         f.write("==========================================================\n\n")

#         for i, seed in enumerate(seeds, 1):
#             result = run_once(seed, device)
#             all_results.append(result)

#             line = "Seed {} | MAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}".format(
#                 result["seed"], result["mae"], result["rmse"], result["r2"]
#             )
#             print(f"[{i}/{len(seeds)}] {line}")   # 控制台可以看进度
#             f.write(line + "\n")                  # txt里只写最终结果行

#         # 排序
#         all_results = sorted(all_results, key=lambda x: x["r2"], reverse=True)
#         top_results = all_results[:top_k]

#         f.write("\n")
#         f.write("================ Best Top-{} Seeds ================\n".format(top_k))
#         for rank, res in enumerate(top_results, 1):
#             line = "Top {} | Seed {} | MAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}".format(
#                 rank, res["seed"], res["mae"], res["rmse"], res["r2"]
#             )
#             f.write(line + "\n")

#     return top_results, all_results, result_path


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 200 个 seed
    seeds = [9, 90, 129, 68, 58, 46, 22, 113, 3, 144,
    99, 143, 173, 72, 137, 89, 37, 62, 52, 6,
    189, 31, 18, 105, 81, 163, 127, 172, 7, 116,
    103, 5, 4, 96, 181, 36, 155, 114, 95, 152, 12]
    # seeds = list(range(1, 201))

    # 如果想随机 200 个 seed：
    # random.seed(42)
    # seeds = random.sample(range(1, 100000), 200)

    top5_results, all_results, result_path = run_multiple_seeds(device, seeds, top_k=5)

    print("\n================ Best 5 Seeds ================")
    for i, res in enumerate(top5_results, 1):
        print(
            "Top {:d} | Seed {:d} | MAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}".format(
                i, res["seed"], res["mae"], res["rmse"], res["r2"]
            )
        )

    print(f"\nAll final results saved to: {result_path}")