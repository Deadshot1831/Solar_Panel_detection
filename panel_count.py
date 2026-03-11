import argparse
import csv
import datetime
import glob
import math
import os
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project modules under `codes/` are importable when running from repo root.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.join(ROOT_DIR, "codes")
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

from data_processing import data_processing_tool_4, prepare_data
from model_list import fast_scnn_2, segnet_0, segnet_1, segnet_3

# =========================
# DEFAULTS
# =========================
DEFAULT_INPUT_IMAGE_DIR = os.path.join(ROOT_DIR, "drone_imgs")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "drone_output-gpu0")
DEFAULT_MODEL_DIR = os.path.join(CODES_DIR, "trained_models")
DEFAULT_H5_MODEL = os.path.join(DEFAULT_MODEL_DIR, "fast_scnn_2.h5")
DEFAULT_METADATA_CSV = os.path.join(DEFAULT_INPUT_IMAGE_DIR, "metadata.csv")
DEFAULT_BATCH_SIZE = 8
DEFAULT_GMAPS_SCALE = 1
DEFAULT_CONF_THRESHOLD = 0.1
DEFAULT_PANEL_AREA_M2 = 12.5

# Filename regex fallback: kmz-zm-19_<lat>_<lon>.png or jpg
FILENAME_RE = re.compile(
    r"^kmz-zm-(?P<zoom>\d+)_(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+)\.(png|jpg|jpeg|JPG)$",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count solar panels from image directory using either Keras (.h5) or YOLO (.pt) model."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_H5_MODEL,
        help="Path to model weights. Supported extensions: .h5/.keras (segmentation), .pt (YOLO).",
    )
    parser.add_argument(
        "--model-type",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Keras segmentation model type (used only for .h5/.keras): 1=fast_scnn_2, 2=segnet_resnet_v2, 3=segnet_4_encoder_decoder, 4=segnet_original.",
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_IMAGE_DIR, help="Directory containing input images.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save outputs.")
    parser.add_argument(
        "--metadata-csv",
        default=DEFAULT_METADATA_CSV,
        help="Optional metadata csv with columns filename,lat,lon,zoom (or file/name + latitude/longitude).",
    )
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold.")
    parser.add_argument("--panel-area-m2", type=float, default=DEFAULT_PANEL_AREA_M2, help="Average panel area in square meters.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for Keras segmentation prediction.")
    parser.add_argument("--gmap-scale", type=int, default=DEFAULT_GMAPS_SCALE, help="Google map scale used in meters-per-pixel estimation.")
    parser.add_argument("--yolo-iou", type=float, default=0.5, help="IoU threshold for YOLO NMS.")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--yolo-device", default=None, help="YOLO device override, e.g. cpu, mps, cuda:0. Default is auto.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size for YOLO inference (image is split before prediction).")
    parser.add_argument("--tile-overlap", type=int, default=0, help="Tile overlap in pixels for YOLO inference.")
    parser.add_argument("--save-tile-preds", action="store_true", help="If set, save per-tile prediction images.")
    return parser.parse_args()


def infer_backend(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    if ext in {".h5", ".keras"}:
        return "keras"
    if ext == ".pt":
        return "yolo"
    raise ValueError(f"Unsupported model extension: {ext}. Use .h5/.keras or .pt")


# =========================
# GEO HELPERS
# =========================
def meters_per_pixel(lat_deg, zoom, scale):
    lat_rad = math.radians(lat_deg)
    return (156543.03392 * math.cos(lat_rad) / (2 ** zoom)) / scale


def parse_from_filename(filename):
    m = FILENAME_RE.match(filename)
    if m:
        zoom = int(m.group("zoom"))
        lat = float(m.group("lat"))
        lon = float(m.group("lon"))
        return lat, lon, zoom
    return None, None, None


def load_metadata(csv_path):
    mapping = {}
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                fname = str(row.get("filename") or row.get("file") or row.get("name"))
                lat = row.get("lat") if "lat" in row.index else row.get("latitude")
                lon = row.get("lon") if "lon" in row.index else row.get("longitude")
                zoom = row.get("zoom")
                if pd.notna(fname) and pd.notna(lat) and pd.notna(lon) and pd.notna(zoom):
                    mapping[os.path.basename(fname)] = (float(lat), float(lon), int(zoom))
        except Exception as exc:
            print("Could not read metadata CSV:", exc)
    return mapping


def get_geo(image_filename, metadata_map):
    if metadata_map is not None and image_filename in metadata_map:
        return metadata_map[image_filename]
    return parse_from_filename(image_filename)


def collect_image_paths(input_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.JPG")
    image_paths = []
    for ext in exts:
        image_paths.extend(sorted(glob.glob(os.path.join(input_dir, ext))))
    return image_paths


# =========================
# KERAS SEGMENTATION PATH
# =========================
def load_keras_model(model_type, model_path, input_shape):
    if model_type == 1:
        model = fast_scnn_2.fast_scnn_v2(input_shape, 1, 2, False)
    elif model_type == 2:
        model = segnet_3.segnet_resnet_v2(input_shape, 1, 2, False)
    elif model_type == 3:
        model = segnet_1.segnet_4_encoder_decoder(input_shape, 1, 2, False)
    elif model_type == 4:
        model = segnet_0.segnet_original(input_shape, 1, 2, False)
    else:
        raise ValueError("Invalid model type")
    model.load_weights(model_path)
    return model


def get_predicted_label_list(sub_imgs, model, threshold, batch_size):
    results_list = []
    n = len(sub_imgs)
    i = 0
    while i < n:
        b = min(batch_size, n - i)
        batch = np.stack(sub_imgs[i : i + b], axis=0).astype(np.float32)
        try:
            preds = model.predict(batch, verbose=0)
        except Exception as exc:
            if batch_size > 1:
                print(f"Warning: model.predict failed with batch_size={batch_size}: {exc}. Retrying with batch_size=1.")
                return get_predicted_label_list(sub_imgs, model, threshold, batch_size=1)
            raise
        for pred in preds:
            binary = np.zeros_like(pred)
            mask = pred[:, :, 1] > threshold
            binary[mask, 1] = 1
            binary[~mask, 0] = 1
            rgb = prepare_data.onehot_to_rgb(binary, prepare_data.id2code)
            results_list.append(rgb)
        i += b
    return results_list


def analyze_components(mask_bool, mpp, panel_area_m2):
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    components = []
    for comp_id in range(1, num_labels):
        pix_area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if mpp is not None:
            area_m2 = pix_area * (mpp ** 2)
            est_panels = area_m2 / panel_area_m2
        else:
            area_m2 = None
            est_panels = None
        components.append(
            {
                "id": int(comp_id),
                "pixel_area": int(pix_area),
                "area_m2": float(area_m2) if area_m2 is not None else None,
                "est_panels": float(est_panels) if est_panels is not None else None,
            }
        )
    return components, max(0, num_labels - 1)


def process_image_keras(image_path, model, args, metadata_map):
    filename = os.path.basename(image_path)
    print("Processing:", filename)

    img = plt.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    h, w = img.shape[:2]

    sub_imgs, padded_img, pw, ph = data_processing_tool_4.get_sub_images(img)
    sub_labels = get_predicted_label_list(sub_imgs, model, args.conf_threshold, args.batch_size)
    full_label = data_processing_tool_4.get_full_predicted_label(ph, pw, sub_labels)

    full_arr = np.asarray(full_label)
    rgb = full_arr[:, :, :3]
    pixels = rgb.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    bg = colors[np.argmax(counts)]
    mask = np.any(rgb != bg, axis=2)
    mask_pixels = int(mask.sum())

    lat, lon, zoom = get_geo(filename, metadata_map)
    if lat is not None and zoom is not None:
        mpp = meters_per_pixel(lat, zoom, args.gmap_scale)
        mask_area_m2 = mask_pixels * (mpp ** 2)
        est_panels_area = mask_area_m2 / args.panel_area_m2
    else:
        mpp = None
        mask_area_m2 = None
        est_panels_area = None

    components, num_components = analyze_components(mask, mpp, args.panel_area_m2)
    est_panels_by_components = None
    if any(c["est_panels"] is not None for c in components):
        est_panels_by_components = sum(c["est_panels"] for c in components if c["est_panels"] is not None)

    base = os.path.splitext(filename)[0]
    os.makedirs(args.output_dir, exist_ok=True)

    txt_path = os.path.join(args.output_dir, base + ".txt")
    with open(txt_path, "w") as file_obj:
        file_obj.write(f"source_file: {filename}\n")
        file_obj.write(f"backend: keras\n")
        file_obj.write(f"model_path: {args.model_path}\n")
        file_obj.write(f"image_width: {w}\n")
        file_obj.write(f"image_height: {h}\n")
        file_obj.write(f"mask_pixels: {mask_pixels}\n")
        file_obj.write(f"num_components: {num_components}\n")
        file_obj.write(f"panel_area_m2: {args.panel_area_m2}\n")
        if mpp is not None:
            file_obj.write(f"lat: {lat}\n")
            file_obj.write(f"lon: {lon}\n")
            file_obj.write(f"zoom: {zoom}\n")
            file_obj.write(f"meters_per_pixel: {mpp:.6f}\n")
            file_obj.write(f"mask_area_m2: {mask_area_m2:.4f}\n")
            file_obj.write(f"estimated_panels_area: {est_panels_area:.2f}\n")
        else:
            file_obj.write("meters_per_pixel: None (lat/zoom not provided)\n")
            file_obj.write("mask_area_m2: None\n")
            file_obj.write("estimated_panels_area: None\n")
        if est_panels_by_components is not None:
            file_obj.write(f"estimated_panels_by_components_sum: {est_panels_by_components:.2f}\n")
        else:
            file_obj.write("estimated_panels_by_components_sum: None\n")
        file_obj.write(f"timestamp_utc: {datetime.datetime.utcnow().isoformat()}Z\n")

    comp_csv_path = os.path.join(args.output_dir, base + "_components.csv")
    with open(comp_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "pixel_area", "area_m2", "est_panels"])
        writer.writeheader()
        for comp in components:
            writer.writerow(comp)

    overlay = data_processing_tool_4.add_transparent_mask(padded_img, full_label, w, h)
    overlay.save(os.path.join(args.output_dir, base + ".png"), "PNG")

    print(
        f"Saved: {base}.png | mask_pixels={mask_pixels} | num_components={num_components} | "
        f"est_panels_area={est_panels_area} | est_panels_by_components={est_panels_by_components}"
    )


# =========================
# YOLO .PT PATH
# =========================
def load_yolo_model(model_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Loading a .pt model requires ultralytics and torch. "
            "Install with: .venv/bin/pip install ultralytics torch"
        ) from exc
    return YOLO(model_path)


def iterate_tiles(img_bgr, tile_size, tile_overlap):
    h, w = img_bgr.shape[:2]
    if tile_size <= 0:
        raise ValueError("--tile-size must be > 0")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("--tile-overlap must be >= 0 and < --tile-size")

    step = tile_size - tile_overlap
    tile_id = 0
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            tile = img_bgr[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            tile_id += 1
            yield tile_id, x0, y0, x1, y1, tile


def compute_iou_xyxy(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_box = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / (union + 1e-9)


def nms_indices(boxes_xyxy, scores, iou_threshold):
    if len(boxes_xyxy) == 0:
        return []

    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores_arr)[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = compute_iou_xyxy(boxes[i], boxes[rest])
        order = rest[ious < iou_threshold]

    return keep


def merge_detections_with_nms(detections, iou_threshold):
    if not detections:
        return []

    kept = []
    class_ids = sorted(set(d["class_id"] for d in detections))
    for cls_id in class_ids:
        cls_idx = [i for i, d in enumerate(detections) if d["class_id"] == cls_id]
        cls_boxes = [
            [detections[i]["x1"], detections[i]["y1"], detections[i]["x2"], detections[i]["y2"]]
            for i in cls_idx
        ]
        cls_scores = [detections[i]["confidence"] for i in cls_idx]
        cls_keep_local = nms_indices(cls_boxes, cls_scores, iou_threshold)
        kept.extend(detections[cls_idx[local_i]] for local_i in cls_keep_local)

    return kept


def process_image_yolo(image_path, yolo_model, args, metadata_map):
    filename = os.path.basename(image_path)
    print("Processing:", filename)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h, w = image_bgr.shape[:2]
    base = os.path.splitext(filename)[0]
    os.makedirs(args.output_dir, exist_ok=True)

    lat, lon, zoom = get_geo(filename, metadata_map)
    if lat is not None and zoom is not None:
        mpp = meters_per_pixel(lat, zoom, args.gmap_scale)
    else:
        mpp = None

    tile_dir = os.path.join(args.output_dir, f"{base}_tiles")
    if args.save_tile_preds:
        os.makedirs(tile_dir, exist_ok=True)

    raw_detections = []
    tiles_processed = 0
    names_map = {}
    for tile_id, x0, y0, x1, y1, tile_bgr in iterate_tiles(image_bgr, args.tile_size, args.tile_overlap):
        tiles_processed += 1
        results = yolo_model.predict(
            source=tile_bgr,
            conf=args.conf_threshold,
            iou=args.yolo_iou,
            imgsz=args.yolo_imgsz,
            device=args.yolo_device,
            save=False,
            verbose=False,
        )
        if not results:
            continue
        result = results[0]
        if hasattr(result, "names") and isinstance(result.names, dict):
            names_map = result.names

        if args.save_tile_preds:
            tile_pred = result.plot()
            tile_name = f"{base}_tile_{tile_id:04d}_x{x0}_y{y0}.png"
            cv2.imwrite(os.path.join(tile_dir, tile_name), tile_pred)

        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(xyxy))
        clss = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.full(len(xyxy), -1)

        for box, conf, cls_id in zip(xyxy, confs, clss):
            tx1, ty1, tx2, ty2 = [float(v) for v in box]
            gx1 = max(0.0, min(float(w), tx1 + x0))
            gy1 = max(0.0, min(float(h), ty1 + y0))
            gx2 = max(0.0, min(float(w), tx2 + x0))
            gy2 = max(0.0, min(float(h), ty2 + y0))
            if gx2 <= gx1 or gy2 <= gy1:
                continue

            cls_int = int(cls_id) if cls_id >= 0 else None
            if isinstance(names_map, dict) and cls_int is not None:
                cls_name = names_map.get(cls_int, str(cls_int))
            else:
                cls_name = str(cls_int) if cls_int is not None else None

            raw_detections.append(
                {
                    "tile_id": tile_id,
                    "tile_x0": x0,
                    "tile_y0": y0,
                    "tile_x1": x1,
                    "tile_y1": y1,
                    "confidence": float(conf),
                    "class_id": cls_int,
                    "class_name": cls_name,
                    "x1": gx1,
                    "y1": gy1,
                    "x2": gx2,
                    "y2": gy2,
                }
            )

    merged_detections = merge_detections_with_nms(raw_detections, args.yolo_iou)

    box_rows = []
    for idx, det in enumerate(merged_detections, start=1):
        bw = max(0.0, det["x2"] - det["x1"])
        bh = max(0.0, det["y2"] - det["y1"])
        pixel_area = bw * bh
        if mpp is not None:
            area_m2 = pixel_area * (mpp ** 2)
            est_panels_box = area_m2 / args.panel_area_m2
        else:
            area_m2 = None
            est_panels_box = None
        box_rows.append(
            {
                "id": idx,
                "tile_id": det["tile_id"],
                "tile_x0": det["tile_x0"],
                "tile_y0": det["tile_y0"],
                "tile_x1": det["tile_x1"],
                "tile_y1": det["tile_y1"],
                "confidence": det["confidence"],
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
                "pixel_area": float(pixel_area),
                "area_m2": float(area_m2) if area_m2 is not None else None,
                "est_panels_box": float(est_panels_box) if est_panels_box is not None else None,
            }
        )

    total_count = len(box_rows)

    txt_path = os.path.join(args.output_dir, base + ".txt")
    with open(txt_path, "w") as file_obj:
        file_obj.write(f"source_file: {filename}\n")
        file_obj.write(f"backend: yolo\n")
        file_obj.write(f"model_path: {args.model_path}\n")
        file_obj.write(f"image_width: {w}\n")
        file_obj.write(f"image_height: {h}\n")
        file_obj.write(f"tile_size: {args.tile_size}\n")
        file_obj.write(f"tile_overlap: {args.tile_overlap}\n")
        file_obj.write(f"tiles_processed: {tiles_processed}\n")
        file_obj.write(f"panel_count_boxes_raw: {len(raw_detections)}\n")
        file_obj.write(f"panel_count_boxes: {total_count}\n")
        file_obj.write(f"panel_area_m2: {args.panel_area_m2}\n")
        file_obj.write("mask_pixels: None (tile detection mode uses box-count)\n")
        file_obj.write("num_components: None\n")
        if mpp is not None:
            file_obj.write(f"lat: {lat}\n")
            file_obj.write(f"lon: {lon}\n")
            file_obj.write(f"zoom: {zoom}\n")
            file_obj.write(f"meters_per_pixel: {mpp:.6f}\n")
            if box_rows:
                total_box_area_m2 = sum(row["area_m2"] for row in box_rows if row["area_m2"] is not None)
                total_est_panels = sum(row["est_panels_box"] for row in box_rows if row["est_panels_box"] is not None)
                file_obj.write(f"mask_area_m2: {total_box_area_m2:.4f}\n")
                file_obj.write(f"estimated_panels_area: {total_est_panels:.2f}\n")
            else:
                file_obj.write("mask_area_m2: None\n")
                file_obj.write("estimated_panels_area: None\n")
        else:
            file_obj.write("meters_per_pixel: None (lat/zoom not provided)\n")
            file_obj.write("mask_area_m2: None\n")
            file_obj.write("estimated_panels_area: None\n")
        file_obj.write("estimated_panels_by_components_sum: None\n")
        file_obj.write(f"timestamp_utc: {datetime.datetime.utcnow().isoformat()}Z\n")

    boxes_csv_path = os.path.join(args.output_dir, base + "_boxes.csv")
    with open(boxes_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "id",
                "tile_id",
                "tile_x0",
                "tile_y0",
                "tile_x1",
                "tile_y1",
                "confidence",
                "class_id",
                "class_name",
                "x1",
                "y1",
                "x2",
                "y2",
                "pixel_area",
                "area_m2",
                "est_panels_box",
            ],
        )
        writer.writeheader()
        for row in box_rows:
            writer.writerow(row)

    comp_csv_path = os.path.join(args.output_dir, base + "_components.csv")
    with open(comp_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "pixel_area", "area_m2", "est_panels"])
        writer.writeheader()
    overlay = image_bgr.copy()
    for row in box_rows:
        x1i, y1i, x2i, y2i = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        label_name = row["class_name"] if row["class_name"] is not None else "panel"
        label = f"{label_name} {row['confidence']:.2f}"
        cv2.putText(overlay, label, (x1i, max(0, y1i - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.output_dir, base + ".png"), overlay)

    print(
        f"Saved: {base}.png | tiles_processed={tiles_processed} | "
        f"panel_count_boxes_raw={len(raw_detections)} | panel_count_boxes={total_count}"
    )


def main():
    args = parse_args()

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT_DIR, model_path)
        args.model_path = model_path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    backend = infer_backend(args.model_path)
    metadata_map = load_metadata(args.metadata_csv)
    image_paths = collect_image_paths(args.input_dir)
    if not image_paths:
        print("No images found in", args.input_dir)
        return

    start = datetime.datetime.now()

    if backend == "keras":
        sample_img = plt.imread(image_paths[0])
        if sample_img.ndim == 3 and sample_img.shape[2] == 4:
            sample_img = sample_img[:, :, :3]
        sub_imgs_sample, _, _, _ = data_processing_tool_4.get_sub_images(sample_img)
        sample_tile_shape = sub_imgs_sample[0].shape

        print("Backend: keras")
        print("Building model with input shape:", sample_tile_shape)
        model = load_keras_model(args.model_type, args.model_path, sample_tile_shape)
        for path in image_paths:
            try:
                process_image_keras(path, model, args, metadata_map=metadata_map)
            except Exception as exc:
                print(f"Error processing {path}: {exc}")
    else:
        print("Backend: yolo")
        yolo_model = load_yolo_model(args.model_path)
        for path in image_paths:
            try:
                process_image_yolo(path, yolo_model, args, metadata_map=metadata_map)
            except Exception as exc:
                print(f"Error processing {path}: {exc}")

    print("Total time:", (datetime.datetime.now() - start))


if __name__ == "__main__":
    main()
