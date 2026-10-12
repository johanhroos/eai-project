"""Utilities for cropping objects from images."""


def pad_bbox(bbox, image_shape, padding=0.1):
    """Add padding to an [x1, y1, x2, y2] bounding box."""
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    pad_x = bbox_width * padding
    pad_y = bbox_height * padding

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(width, int(x2 + pad_x))
    y2 = min(height, int(y2 + pad_y))

    return x1, y1, x2, y2


def crop_bbox(image, bbox, padding=0.1):
    """Crop one padded bounding box."""
    x1, y1, x2, y2 = pad_bbox(bbox, image.shape, padding)
    return image[y1:y2, x1:x2]


def crop_bboxes(image, bboxes, padding=0.1):
    """Crop multiple padded bounding boxes."""
    return [crop_bbox(image, bbox, padding) for bbox in bboxes]


def square_bbox(bbox, image_shape):
    """Grow an [x1, y1, x2, y2] box to a square around its centre, shifted to stay in the image where it fits."""
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]
    side = min(max(x2 - x1, y2 - y1), width, height)
    x1 = min(max(0, (x1 + x2 - side) // 2), width - side)
    y1 = min(max(0, (y1 + y2 - side) // 2), height - side)
    return x1, y1, x1 + side, y1 + side
