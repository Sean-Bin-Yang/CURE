import torch
import torch.nn as nn
import torch.nn.functional as F
from parse_args import args


# ============================================================
# Utils
# ============================================================
def ensure_3d(x):
    if x.dim() == 2:
        return x.unsqueeze(0)
    return x


def drop_batch_if_present(x):
    if x.dim() == 3 and x.size(0) == 1:
        return x.squeeze(0)
    return x


def normalize_adj(A):
    if A is None:
        return None
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    A = A + I
    D = A.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return A / D


# ============================================================
# Shared MLP
# ============================================================
class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(0.3, inplace=True),
        )
        self.output = None

    def forward(self, x):
        self.output = self.model(x)
        return self.output

    def out_feature(self):
        return self.output


# ============================================================
# Intra-view: Graph-guided local encoder
# ============================================================
class GraphGuidedIntraBlock(nn.Module):
    def __init__(self, input_dim, nhead, dropout, dim_feedforward=2048):
        super().__init__()
        self.graph_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.self_attn = nn.MultiheadAttention(
            input_dim, nhead, dropout=dropout, batch_first=True, bias=True
        )
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm0 = nn.LayerNorm(input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, A):
        # src: [B, N, D], A: [N, N]
        h_graph = torch.matmul(A, self.graph_proj(src))
        src = self.norm0(src + self.dropout(h_graph))

        src2, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(src2))

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class GraphGuidedIntraAFL(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.num_block = args.NO_IntraAFL
        self.blocks = nn.ModuleList([
            GraphGuidedIntraBlock(
                input_dim=input_dim,
                nhead=args.NO_head,
                dropout=args.dropout
            ) for _ in range(self.num_block)
        ])
        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x, A):
        A = normalize_adj(A)
        out = ensure_3d(x)
        for block in self.blocks:
            out = block(out, A)
        out = drop_batch_if_present(out)
        return self.fc(out)


# ============================================================
# Region-level refinement
# ============================================================
class RegionFusionBlock(nn.Module):
    def __init__(self, input_dim, nhead, dropout, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            input_dim, nhead, dropout=dropout, batch_first=True, bias=True
        )
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(src2))

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class RegionFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.num_block = args.NO_RegionFusion
        self.blocks = nn.ModuleList([
            RegionFusionBlock(
                input_dim=input_dim,
                nhead=args.NO_head,
                dropout=args.dropout
            ) for _ in range(self.num_block)
        ])
        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = ensure_3d(x)
        for block in self.blocks:
            out = block(out)
        out = drop_batch_if_present(out)
        return self.fc(out)


# ============================================================
# Inter-view block
# ============================================================
class InterAFLBlock(nn.Module):
    def __init__(self, d_model, S):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        # queries: [N, V, D]
        attn = self.mk(queries)  # [N, V, S]
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True).clamp(min=1e-12)
        out = self.mv(attn)      # [N, V, D]
        return out


# ============================================================
# Inter-view: confounder-aware interaction
# ============================================================
class ConfounderAwareInterAFL(nn.Module):
    def __init__(self, input_dim, d_m, hidden=128):
        super().__init__()
        self.input_dim = input_dim
        self.num_block = args.NO_InterAFL

        self.confounder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(hidden, input_dim),
        )

        self.proj_list = nn.ModuleList()
        self.blocks = nn.ModuleList(
            [InterAFLBlock(input_dim, d_m) for _ in range(self.num_block)]
        )
        self.fc = DeepFc(input_dim, input_dim)

        self.cached_c = None
        self.cached_r_views = None

    def _ensure_proj(self, V, d_model, device):
        if len(self.proj_list) == V:
            return
        self.proj_list = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(V)]
        ).to(device)

    def forward(self, z_views):
        if z_views.dim() != 3:
            raise ValueError(f"Expected z_views [V,N,D], got {tuple(z_views.shape)}")

        V, N, D = z_views.shape
        self._ensure_proj(V, D, z_views.device)

        z_mean = z_views.mean(dim=0)  # [N, D]
        c = self.confounder(z_mean)   # [N, D]

        r_list = []
        for v in range(V):
            r_v = z_views[v] - self.proj_list[v](c)
            r_list.append(r_v)
        r_views = torch.stack(r_list, dim=0)  # [V, N, D]

        self.cached_c = c
        self.cached_r_views = r_views

        out = r_views.transpose(0, 1)  # [N, V, D]
        for block in self.blocks:
            out = block(out)
        out = self.fc(out)
        out = out.transpose(0, 1)      # [V, N, D]
        return out


# ============================================================
# Hierarchical graph context
# ============================================================
class HierarchicalGraphContext(nn.Module):
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.local_proj = nn.Linear(d_model, d_model, bias=False)
        self.global_proj = nn.Linear(d_model, d_model, bias=False)

        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, hidden),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(hidden, d_model),
            nn.LeakyReLU(0.3, inplace=True),
        )

    def forward(self, x, A_local, A_global):
        h_local = torch.matmul(A_local, self.local_proj(x))
        h_global = torch.matmul(A_global, self.global_proj(x))
        h = torch.cat([x, h_local, h_global], dim=-1)
        return self.fuse(h)


# ============================================================
# Fusion: hierarchical graph-aware residual fusion
# ============================================================
class HierarchicalResidualFusion(nn.Module):
    def __init__(self, d_model, hidden=128, use_confounder_skip=True):
        super().__init__()
        self.use_confounder_skip = use_confounder_skip

        self.graph_context = HierarchicalGraphContext(d_model, hidden)

        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(hidden, 1)
        )

        self.post = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.3, inplace=True),
        )

        self.last_alpha = None

    def forward(self, r_views, c=None, A_local=None, A_global=None):
        if r_views.dim() != 3:
            raise ValueError(f"Expected r_views [V,N,D], got {tuple(r_views.shape)}")
        if A_local is None or A_global is None:
            raise ValueError("A_local and A_global must be provided")

        h_list = []
        for v in range(r_views.size(0)):
            h_v = self.graph_context(r_views[v], A_local, A_global)
            h_list.append(h_v)
        h_views = torch.stack(h_list, dim=0)  # [V, N, D]

        scores = self.score(h_views)          # [V, N, 1]
        alpha = F.softmax(scores, dim=0)
        self.last_alpha = alpha.detach()

        fused = (alpha * h_views).sum(dim=0)

        if self.use_confounder_skip and c is not None:
            fused = fused + c

        return self.post(fused)


# ============================================================
# Full model
# ============================================================
class CauFusion(nn.Module):
    def __init__(
        self,
        poi_dim,
        landUse_dim,
        input_dim,
        output_dim,
        d_prime,
        d_m,
        c,
        A_local=None,
        A_global=None,
        A_poi=None,
        A_landuse=None,
        A_mob=None
    ):
        super().__init__()
        self.input_dim = input_dim

        # Input projections
        self.densePOI2 = nn.Linear(poi_dim, input_dim)
        self.denseLandUse3 = nn.Linear(landUse_dim, input_dim)
        self.denseMob = nn.Linear(args.region_num, input_dim)

        # Intra-view encoders
        self.encoderPOI = GraphGuidedIntraAFL(input_dim)
        self.encoderLandUse = GraphGuidedIntraAFL(input_dim)
        self.encoderMob = GraphGuidedIntraAFL(input_dim)

        # Inter-view + fusion
        self.interViewEncoder = ConfounderAwareInterAFL(
            input_dim=input_dim,
            d_m=d_m,
            hidden=max(64, d_prime)
        )
        self.causalFusionLayer = HierarchicalResidualFusion(
            d_model=input_dim,
            hidden=max(64, d_prime),
            use_confounder_skip=True
        )
        self.regionFusionLayer = RegionFusion(input_dim)

        # Output head
        self.fc = DeepFc(input_dim, output_dim)

        # Mixing weights
        self.para1 = nn.Parameter(torch.tensor(0.1))
        self.para2 = nn.Parameter(torch.tensor(0.9))

        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)

        self.decoder_s = nn.Linear(output_dim, output_dim)
        self.decoder_t = nn.Linear(output_dim, output_dim)
        self.decoder_p = nn.Linear(output_dim, output_dim)
        self.decoder_l = nn.Linear(output_dim, output_dim)

        self.feature = None

        # Graph buffers
        self.register_buffer("A_local", normalize_adj(A_local) if A_local is not None else None)
        self.register_buffer("A_global", normalize_adj(A_global) if A_global is not None else None)
        self.register_buffer("A_poi", normalize_adj(A_poi) if A_poi is not None else None)
        self.register_buffer("A_landuse", normalize_adj(A_landuse) if A_landuse is not None else None)
        self.register_buffer("A_mob", normalize_adj(A_mob) if A_mob is not None else None)

    def set_graphs(self, A_local, A_global, A_poi=None, A_landuse=None, A_mob=None):
        self.A_local = normalize_adj(A_local)
        self.A_global = normalize_adj(A_global)
        self.A_poi = normalize_adj(A_poi if A_poi is not None else A_global)
        self.A_landuse = normalize_adj(A_landuse if A_landuse is not None else A_local)
        self.A_mob = normalize_adj(A_mob if A_mob is not None else A_local)

    def _check_graphs(self):
        if self.A_local is None or self.A_global is None:
            raise ValueError("A_local and A_global are not set. Call set_graphs(...) or pass them in initialization.")
        if self.A_poi is None:
            self.A_poi = self.A_global
        if self.A_landuse is None:
            self.A_landuse = self.A_local
        if self.A_mob is None:
            self.A_mob = self.A_local

    def forward(self, x):
        # x: (poi_emb, landUse_emb, mob_emb)
        poi_emb, landUse_emb, mob_emb = x
        self._check_graphs()

        # Project raw features
        poi_emb = self.dropout(self.activation(self.densePOI2(poi_emb)))
        landUse_emb = self.dropout(self.activation(self.denseLandUse3(landUse_emb)))
        mob_emb = self.dropout(self.activation(self.denseMob(mob_emb)))

        # Intra-view encoding
        poi_emb = self.encoderPOI(poi_emb, self.A_poi)
        landUse_emb = self.encoderLandUse(landUse_emb, self.A_landuse)
        mob_emb = self.encoderMob(mob_emb, self.A_mob)

        # Stack views
        out = torch.stack([poi_emb, landUse_emb, mob_emb], dim=0)  # [V, N, D]

        # Confounder-aware inter-view interaction
        out_inter = self.interViewEncoder(out)                     # [V, N, D]
        c = self.interViewEncoder.cached_c
        residual_views = self.interViewEncoder.cached_r_views

        # Mix interacted residuals and raw residuals
        p1 = self.para1 / (self.para1 + self.para2 + 1e-12)
        p2 = self.para2 / (self.para1 + self.para2 + 1e-12)
        out = out_inter * p2 + residual_views * p1

        # Hierarchical residual fusion
        temp_out = self.causalFusionLayer(
            out,
            c=c,
            A_local=self.A_local,
            A_global=self.A_global
        )  # [N, D]

        # Region refinement
        temp_out = temp_out.unsqueeze(0)          # [1, N, D]
        temp_out = self.regionFusionLayer(temp_out)

        # Final embedding
        out = self.fc(temp_out)
        self.feature = out

        # Decoders
        out_s = self.decoder_s(out)
        out_t = self.decoder_t(out)
        out_p = self.decoder_p(out)
        out_l = self.decoder_l(out)
        return out_s, out_t, out_p, out_l

    def out_feature(self):
        return self.feature
