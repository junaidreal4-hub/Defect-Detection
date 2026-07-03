import numpy as np
import torch
import torch.nn as nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

# Cap on the number of patches fed to coreset selection. The greedy k-center
# pass is O(pool * kept), so bounding the pool keeps memory-bank construction
# tractable on CPU while still sampling the feature distribution densely.
CORESET_POOL_SIZE = 10000


class PatchCore(nn.Module):
    def __init__(self, device: str = "cpu", coreset_ratio: float = 0.1):
        super().__init__()
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.memory_bank = None
        self.threshold = None
        self._features = {}

        # layer2 + layer3 only: layer1 is too generic and blows up the patch
        # count, layer4 is too semantic to localise defects. Their concatenation
        # is the mid-level representation PatchCore relies on.
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.eval()
        backbone.layer2.register_forward_hook(self._save_output("layer2"))
        backbone.layer3.register_forward_hook(self._save_output("layer3"))
        self.backbone = backbone.to(device)

        for param in self.backbone.parameters():
            param.requires_grad = False

    def _save_output(self, name):
        def hook(_module, _input, output):
            self._features[name] = output
        return hook

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.backbone(x)

        layer2, layer3 = self._features["layer2"], self._features["layer3"]
        layer3 = nn.functional.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat([layer2, layer3], dim=1)

    def _reshape_to_patches(self, features: torch.Tensor):
        _, C, H, W = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(-1, C)
        return patches.cpu().numpy(), H, W

    def build_memory_bank(self, features_list: list):
        patches = np.vstack(features_list)
        n_keep = max(1, int(len(patches) * self.coreset_ratio))
        selected = self._coreset_indices(patches, n_keep)

        self.memory_bank = torch.tensor(
            patches[selected], dtype=torch.float32, device=self.device
        )
        return self.memory_bank

    def _coreset_indices(self, features: np.ndarray, n_samples: int) -> np.ndarray:
        """Greedy farthest-point (k-center) selection over a capped random pool.

        Each step keeps the point maximally distant from everything chosen so
        far, spreading the memory bank across the feature manifold far better
        than uniform sampling would.
        """
        if len(features) > CORESET_POOL_SIZE:
            # Fixed seed so a rebuilt memory bank is reproducible; without it the
            # random pool draw swings image-level AU-ROC by ~0.03 between runs.
            rng = np.random.default_rng(0)
            pool_idx = rng.choice(len(features), CORESET_POOL_SIZE, replace=False)
        else:
            pool_idx = np.arange(len(features))
        pool = features[pool_idx]

        n_samples = min(n_samples, len(pool))
        selected = [0]
        min_dists = np.full(len(pool), np.inf)
        for _ in range(n_samples - 1):
            diff = pool - pool[selected[-1]]
            dists = np.einsum("ij,ij->i", diff, diff)  # squared L2; sqrt is monotonic
            min_dists = np.minimum(min_dists, dists)
            selected.append(int(np.argmax(min_dists)))

        return pool_idx[selected]

    def score(self, x: torch.Tensor):
        features = self.extract_features(x.to(self.device))
        patches, H, W = self._reshape_to_patches(features)

        patches = torch.tensor(patches, dtype=torch.float32, device=self.device)
        patch_scores = torch.cdist(patches, self.memory_bank).min(dim=1).values

        heatmap = patch_scores.cpu().numpy().reshape(H, W)
        return float(heatmap.max()), heatmap
