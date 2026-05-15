import logging
import numpy as np
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)

class HullPlugin:
    """
    Extracts per-object features from a 2D segmentation array based on a 
    specific target label.
    """

    def run(self, segmentation: np.ndarray, target_label: int) -> list[dict]:
        """
        Extracts region features (centroid, area, bbox, convex_area) for the 
        target label.

        Args:
            segmentation (np.ndarray): 2D array of shape (256, 256).
            target_label (int): Label to extract features for (0, 1, or 2).

        Returns:
            list[dict]: List of feature dictionaries for each object found.
        """
        if not isinstance(segmentation, np.ndarray) or segmentation.shape != (256, 256):
            raise ValueError("Segmentation must be a numpy array of shape (256, 256).")
        
        if target_label not in (0, 1, 2):
            raise ValueError("Target label must be 0, 1, or 2.")

        logger.info("Starting HullPlugin.run with segmentation shape: %s for target_label: %s", segmentation.shape, target_label)

        # Create binary mask for the target label
        target_mask = (segmentation == target_label).astype(int)
        
        # Label connected components
        logger.debug("Labeling connected components...")
        labeled_mask = label(target_mask)
        
        # Extract features
        regions = regionprops(labeled_mask)
        features = []
        
        for region in regions:
            features.append({
                'centroid': region.centroid,
                'area': region.area,
                'bbox': region.bbox,
                'convex_area': region.area_convex
            })
        
        logger.info("Extracted %d feature regions for target_label: %s.", len(features), target_label)
            
        return features
