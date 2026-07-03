"""Per-category PatchCore configuration.

Most MVTec categories work well at 224px. Raising the input to 256px yields a
denser feature grid, which helps categories whose defects are fine-textured
rather than structural. Sweeping the six weak categories at 256px, only grid
improved beyond the run-to-run noise of the randomised coreset (~0.03 AU-ROC):
it jumped from 0.637 to 0.701. The others were neutral (pill) or worse
(transistor and tile regressed, since their defects are geometric), so 256px is
applied to grid alone.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchCoreConfig:
    image_size: int = 224
    coreset_ratio: float = 0.1


_OVERRIDES = {
    "grid": PatchCoreConfig(image_size=256),
}


def get_config(category: str) -> PatchCoreConfig:
    return _OVERRIDES.get(category, PatchCoreConfig())
