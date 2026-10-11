
"""Utilities for cropping objects from images."""

import math


def pad_bbox(bbox, image_shape, padding=0.1):
    """Add padding to an [x1, y1, x2, y2] bounding box."""
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    pad_x = bbox_width * padding
    pad_y = bbox_height * padding

    x1 = max(0, math.floor(x1 - pad_x))
    y1 = max(0, math.floor(y1 - pad_y))
    x2 = min(width, math.ceil(x2 + pad_x))
    y2 = min(height, math.ceil(y2 + pad_y))

    return x1, y1, x2, y2


def crop_bbox(image, bbox, padding=0.1):
    """Crop one padded bounding box."""
    x1, y1, x2, y2 = pad_bbox(bbox, image.shape, padding)
    return image[y1:y2, x1:x2]


def crop_bboxes(image, bboxes, padding=0.1):
    """Crop multiple padded bounding boxes."""

    crops = []

    for sam_bbox in bboxes:
        x1, y1, x2, y2 = sam_bbox
        bbox = (x1, y1, x2 + 1, y2 + 1)
        crops.append(crop_bbox(image, bbox, padding))
    
    return crops
