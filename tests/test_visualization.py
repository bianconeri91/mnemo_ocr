import cv2
import numpy as np

from src.visualization import (
    build_color_mask,
    extract_rois,
    save_mask,
    save_rois,
)


def test_mask_and_rois_are_saved(tmp_path):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 30), (150, 50), (0, 255, 0), -1)
    color_ranges = {
        "green": ([50, 80, 80], [80, 255, 255]),
    }

    mask = build_color_mask(img, color_ranges)
    rois, positions = extract_rois(img, mask)

    mask_path = save_mask(mask, "test image", tmp_path / "masks")
    roi_paths = save_rois(
        rois,
        positions,
        "test image",
        tmp_path / "rois",
    )

    assert mask_path.exists()
    assert mask_path.name == "test_image_mask.png"
    assert len(roi_paths) == 1
    assert roi_paths[0].exists()
    assert cv2.imread(str(roi_paths[0])).shape[:2] == (17, 131)
