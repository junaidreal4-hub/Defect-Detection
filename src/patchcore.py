import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import numpy as np


class PatchCore(nn.Module):
    def __init__(self, device: str = "cpu", coreset_ratio: float = 0.1):
        super().__init__()
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.memory_bank = None
        self.layer2_output = None
        self.layer3_output = None

        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.eval()
        backbone.layer2.register_forward_hook(self._hook("layer2"))
        backbone.layer3.register_forward_hook(self._hook("layer3"))
        self.backbone = backbone.to(device)

        for param in self.backbone.parameters():
            param.requires_grad = False

    def _hook(self, name):
        def hook_fn(module, input, output):
            if name == "layer2":
                self.layer2_output = output
            elif name == "layer3":
                self.layer3_output = output
        return hook_fn

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.backbone(x)

        layer3_up = nn.functional.interpolate(
            self.layer3_output,
            size=self.layer2_output.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat([self.layer2_output, layer3_up], dim=1)
        return features

    def _reshape_to_patches(self, features: torch.Tensor):
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)
        return features.reshape(-1, C).cpu().numpy(), H, W

    def build_memory_bank(self, features_list: list):
        all_patches = np.vstack(features_list)  # (N, C)

        n_keep = max(1, int(len(all_patches) * self.coreset_ratio))
        selected_idx = self._batched_coreset(all_patches, n_keep)

        self.memory_bank = all_patches[selected_idx]
        self.memory_bank_tensor = torch.tensor(
            self.memory_bank, dtype=torch.float32
        ).to(self.device)
        print(f"Memory bank built: {self.memory_bank_tensor.shape[0]} patches retained.")

    def _batched_coreset(self, features: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Greedy farthest-point coreset but vectorized in batches.
        Fast enough on CPU — avoids the row-by-row loop that caused the hang.
        """
        # Work on a random subset to keep it manageable (max 10k points)
        max_pool = 10000
        if len(features) > max_pool:
            pool_idx = np.random.choice(len(features), max_pool, replace=False)
            pool = features[pool_idx]
        else:
            pool_idx = np.arange(len(features))
            pool = features

        n_samples = min(n_samples, len(pool))
        selected = [0]
        # min distance from each point to the closest already-selected point
        min_dists = np.full(len(pool), np.inf)

        for _ in range(n_samples - 1):
            last = pool[selected[-1]]  # (C,)

            # Vectorized: compute distance from ALL points to just the last selected
            diff = pool - last          # (N, C)
            dists = np.einsum('ij,ij->i', diff, diff)  # squared L2, no sqrt needed

            min_dists = np.minimum(min_dists, dists)
            selected.append(int(np.argmax(min_dists)))

        return pool_idx[np.array(selected)]
    def score(self, x: torch.Tensor):
        features = self.extract_features(x.to(self.device))
        patches, H, W = self._reshape_to_patches(features)

        patches_t = torch.tensor(patches, dtype=torch.float32).to(self.device)
        dists = torch.cdist(patches_t, self.memory_bank_tensor, p=2)
        patch_scores = dists.min(dim=1).values.cpu().numpy()

        heatmap = patch_scores.reshape(H, W)
        anomaly_score = float(patch_scores.max())

        return anomaly_score, heatmap