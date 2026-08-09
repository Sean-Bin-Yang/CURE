# parse_args.py
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # ----------------------- Basic ------------------------
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--city", type=str, default="NY", choices=["NY", "Chi", "SF"],
                        help="City name: NY / Chi / SF")
    parser.add_argument("--task", type=str, default="checkIn", choices=["crime", "checkIn", "serviceCall"],
                        help="Downstream task: crime / checkIn / serviceCall")

    # ----------------------- File ------------------------
    parser.add_argument("--mobility_dist", type=str, default="/mob_dist.npy")
    parser.add_argument("--POI_dist", type=str, default="/poi_dist.npy")
    parser.add_argument("--landUse_dist", type=str, default="/landUse_dist.npy")
    parser.add_argument("--mobility_adj", type=str, default="/mob-adj.npy")
    parser.add_argument("--POI_simi", type=str, default="/poi_simi.npy")
    parser.add_argument("--landUse_simi", type=str, default="/landUse_simi.npy")

    # ----------------------- Model / Train ------------------------
    parser.add_argument("--embedding_size", type=int, default=144,
                        help="Output embedding dim (d). e.g., 36 64 72 96 144 288")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ----------------------- Router params ------------------------
    parser.add_argument("--router_topk", type=int, default=2)
    parser.add_argument("--router_hidden", type=int, default=128)
    parser.add_argument("--router_temp", type=float, default=1.0)

    # routing regularization
    parser.add_argument("--lambda_moe", type=float, default=0.01)
    parser.add_argument("--route_objective", type=str, default="sharp",
                        choices=["sharp", "uniform"])

    # ----------------------- Concat ablation params ------------------------
    # parser.add_argument("--fusion_mode", type=str, default="router",
    #                     choices=["router", "concat"],
    #                     help="Fusion type: router (MoE) or concat (ablation)")
    
    parser.add_argument(
    "--fusion_mode",
    type=str,
    default="attention_gated",
    choices=[
        "attention_gated",
        "spatial_attention",
        "dense_gated",
        "soft_routing",
        "expert_choice",
        "router",
        "concat",
    ],
)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--concat_use_mlp", action="store_true",
                        help="Use MLP projection after concatenation")
    parser.add_argument("--concat_hidden", type=int, default=None,
                        help="Hidden dim for concat MLP projection (if concat_use_mlp)")

    # ----------------------- PID-based MSE ------------------------
    parser.add_argument("--use_pid_mse", action="store_true")
    parser.add_argument("--pid_alpha", type=float, default=1.0)
    parser.add_argument("--pid_beta", type=float, default=0.01)
    parser.add_argument("--pid_gamma", type=float, default=0.05)
    parser.add_argument("--pid_leak", type=float, default=0.99)
    parser.add_argument("--pid_dim", type=int, default=1)

    # ----------------------- Auto weight multi-loss ------------------------
    parser.add_argument("--auto_weight", action="store_true")

    # ----------------------- (Optional) Causal ------------------------
    parser.add_argument("--lambda_causal", type=float, default=0.006,
                        help="weight for causal deconfounding loss (if used)")

    # ----------------------- FLOPs / Params report ------------------------
    parser.add_argument("--verbose_flops", action="store_true",
                        help="Whether to print per-module flop table")
    parser.add_argument("--save_flops_report", action="store_true", default=True,
                        help="Whether to save FLOPs report to file")

    # ----------------------- Profiling (time/memory) ------------------------
    parser.add_argument("--profile_warmup", type=int, default=5,
                        help="Ignore first K epochs when reporting avg/p95 epoch time")

    # ----------------------- Robustness: Missing view ------------------------
    # Example:
    #   --robust_eval --missing_view poi --missing_p_list 0.1 0.3 0.5 1.0 --missing_trials 5
    parser.add_argument("--robust_eval", action="store_true",
                        help="Run missing-view robustness eval (no training), or run after training in robust script.")
    parser.add_argument("--missing_view", type=str, default="random_one",
                        choices=["poi", "land", "mob", "random_one"],
                        help="Which view to mask in missing-view test")
    parser.add_argument("--missing_p", type=float, default=0.0,
                    help="Fraction of regions to mask (0=no mask, 1=mask all).")
    parser.add_argument("--missing_trials", type=int, default=5,
                        help="How many random trials (different masks).")
    
    # ---- Noisy-view robustness ----
    parser.add_argument("--noise_eval", action="store_true",
                        help="Run robustness eval with one noisy view after training.")
    parser.add_argument("--noisy_view", type=str, default="poi",
                        choices=["poi", "land", "mob"],
                        help="Which view to corrupt in noisy-view test.")
    parser.add_argument("--noise_type", type=str, default="gaussian",
                        choices=["gaussian", "mul_gaussian", "col_shuffle", "row_shuffle", "feat_dropout"],
                        help="Type of noise/corruption to inject.")
    parser.add_argument("--noise_level", type=float, default=0.0,
                        help="Noise intensity (sigma for gaussian; prob for feat_dropout).")
    parser.add_argument("--noise_trials", type=int, default=5,
                        help="Number of random trials for the same noisy setting.")
    parser.add_argument("--noise_clamp_nonneg", action="store_true", default=True,
                        help="Clamp features to be non-negative after noise.")
    parser.add_argument("--noise_renorm_rows", action="store_true", default=False,
                        help="Row-normalize after noise (recommended for POI/Land if they are distributions).")

    #-----------Magic----------------__#
    parser.add_argument("--num_granularities", type=int, default=3,
                    help="Number of spatial granularities, e.g., micro/meso/macro")

    parser.add_argument("--num_tasks", type=int, default=3,
                        help="Number of downstream tasks")

    parser.add_argument("--lambda_task_div", type=float, default=0.01,
                        help="Weight for task-granularity diversity regularization")

    parser.add_argument("--lambda_gran_smooth", type=float, default=0.01,
                        help="Weight for granularity smoothness regularization")

    parser.add_argument("--control_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_control_entropy", type=float, default=0.0)
    parser.add_argument("--lambda_control_smooth", type=float, default=0.0)

    parser.add_argument("--loss_balancer", type=str, default="pid",
                    choices=["pid", "none", "gradnorm", "pcgrad", "mgda"])
    parser.add_argument("--gradnorm_alpha", type=float, default=1.5)
    parser.add_argument("--pcgrad_reduction", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--mgda_max_iter", type=int, default=50)


    return parser


# -------- City configs --------
CITY_CFG = {
    "NY": dict(
        data_path="./data_NY",
        POI_dim=26, landUse_dim=11, region_num=180,
        NO_IntraAFL=3, NO_InterAFL=3, NO_RegionFusion=3,
        NO_head=4, d_prime=64, d_m=72, c=32
    ),
    "Chi": dict(
        data_path="./data_Chi",
        POI_dim=26, landUse_dim=12, region_num=77,
        NO_IntraAFL=1, NO_InterAFL=2, NO_RegionFusion=3,
        NO_head=1, d_prime=32, d_m=36, c=32
    ),
    "SF": dict(
        data_path="./data_SF",
        POI_dim=26, landUse_dim=23, region_num=175,
        NO_IntraAFL=3, NO_InterAFL=2, NO_RegionFusion=3,
        NO_head=4, d_prime=64, d_m=72, c=32
    )
}


def apply_city_config(args):
    if args.city not in CITY_CFG:
        raise ValueError(f"Unknown city: {args.city}. Choose from {list(CITY_CFG.keys())}")

    # apply city-specific overrides
    for k, v in CITY_CFG[args.city].items():
        setattr(args, k, v)

    # concat_hidden default (only when concat_use_mlp and concat_hidden is None)
    # IMPORTANT: in your MoERoutingFusion, input_dim == region_num (not embedding_size)
    if args.concat_use_mlp and args.concat_hidden is None:
        args.concat_hidden = args.num_views * args.region_num

    return args


parser = get_parser()
args = parser.parse_args()
args = apply_city_config(args)
