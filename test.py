import argparse
import csv
import datetime
import glob
import math
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project modules under `codes/` are importable when running from repo root.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.join(ROOT_DIR, "codes")
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

# =========================
# DEFAULTS
# =========================
DEFAULT_INPUT_IMAGE_DIR = os.path.join(ROOT_DIR, "drone_imgs")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "drone_output-gpu0")
DEFAULT_MODEL_DIR = os.path.join(CODES_DIR, "trained_models")
DEFAULT_PT_MODEL = os.path.join(DEFAULT_MODEL_DIR, "runs_detect_runs_Solar_panel_v12_weights_best (1).pt")
DEFAULT_METADATA_CSV = os.path.join(DEFAULT_INPUT_IMAGE_DIR, "metadata.csv")
DEFAULT_GMAPS_SCALE = 1
DEFAULT_CONF_THRESHOLD = 0.1
DEFAULT_PANEL_AREA_M2 = 12.5

# Filename regex fallback: kmz-zm-19_<lat>_<lon>.png or jpg
FILENAME_RE = re.compile(
    r"^kmz-zm-(?P<zoom>\d+)_(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+)\.(png|jpg|jpeg|JPG)$",
    re.IGNORECASE,
)
XMP_GPS_LAT_RE = re.compile(r'drone-dji:GpsLatitude=\"([+-]?\d+(?:\.\d+)?)\"')
XMP_GPS_LON_RE = re.compile(r'drone-dji:GpsLongitude=\"([+-]?\d+(?:\.\d+)?)\"')
XMP_REL_ALT_RE = re.compile(r'drone-dji:RelativeAltitude=\"([+-]?\d+(?:\.\d+)?)\"')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count solar panels from image directory using YOLO (.pt) model."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_PT_MODEL,
        help="Path to YOLO .pt model weights.",
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
    parser.add_argument("--gmap-scale", type=int, default=DEFAULT_GMAPS_SCALE, help="Google map scale used in meters-per-pixel estimation.")
    parser.add_argument("--yolo-iou", type=float, default=0.5, help="IoU threshold for YOLO NMS.")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--yolo-device", default=None, help="YOLO device override, e.g. cpu, mps, cuda:0. Default is auto.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size for YOLO inference (image is split before prediction).")
    parser.add_argument("--tile-overlap", type=int, default=0, help="Tile overlap in pixels for YOLO inference.")
    parser.add_argument("--save-tile-preds", action="store_true", help="If set, save per-tile prediction images.")
    parser.add_argument(
        "--line-overlap-keep-ratio",
        type=float,
        default=1.0,
        help="For overlap skipping, keep this fraction of computed overlap on the overlap side (1.0 = full overlap skip).",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.02,
        help="Treat overlaps smaller than this ratio (of current image projection) as zero overlap.",
    )
    return parser.parse_args()


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


def dms_tuple_to_decimal(dms_tuple, ref):
    if dms_tuple is None or len(dms_tuple) < 3:
        return None
    deg = float(dms_tuple[0])
    minutes = float(dms_tuple[1])
    seconds = float(dms_tuple[2])
    val = deg + (minutes / 60.0) + (seconds / 3600.0)
    if ref in {"S", "W"}:
        val = -val
    return val


def extract_embedded_drone_metadata(image_path):
    metadata = {
        "lat": None,
        "lon": None,
        "relative_altitude_m": None,
        "focal_length_35mm": None,
        "footprint_width_m": None,
        "footprint_height_m": None,
    }

    try:
        raw = open(image_path, "rb").read().decode("latin-1", errors="ignore")
        mlat = XMP_GPS_LAT_RE.search(raw)
        mlon = XMP_GPS_LON_RE.search(raw)
        malt = XMP_REL_ALT_RE.search(raw)
        if mlat and mlon:
            metadata["lat"] = float(mlat.group(1))
            metadata["lon"] = float(mlon.group(1))
        if malt:
            metadata["relative_altitude_m"] = float(malt.group(1))
    except Exception:
        pass

    try:
        exif = Image.open(image_path).getexif()
        exif_ifd = exif.get_ifd(34665) if 34665 in exif else {}
        gps_ifd = exif.get_ifd(34853) if 34853 in exif else {}
        focal35 = exif_ifd.get(41989)
        if focal35 is not None:
            metadata["focal_length_35mm"] = float(focal35)
        if metadata["lat"] is None and gps_ifd:
            lat_ref = gps_ifd.get(1)
            lat_dms = gps_ifd.get(2)
            lon_ref = gps_ifd.get(3)
            lon_dms = gps_ifd.get(4)
            lat = dms_tuple_to_decimal(lat_dms, lat_ref) if lat_ref and lat_dms else None
            lon = dms_tuple_to_decimal(lon_dms, lon_ref) if lon_ref and lon_dms else None
            if lat is not None and lon is not None:
                metadata["lat"] = lat
                metadata["lon"] = lon
    except Exception:
        pass

    if metadata["relative_altitude_m"] is not None and metadata["focal_length_35mm"] is not None:
        h = metadata["relative_altitude_m"]
        f35 = metadata["focal_length_35mm"]
        hfov = 2.0 * math.atan(36.0 / (2.0 * f35))
        vfov = 2.0 * math.atan(24.0 / (2.0 * f35))
        metadata["footprint_width_m"] = 2.0 * h * math.tan(hfov / 2.0)
        metadata["footprint_height_m"] = 2.0 * h * math.tan(vfov / 2.0)
    return metadata


def image_center_global_m(lat, lon, ref_lat):
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(ref_lat))
    return lon * meters_per_deg_lon, lat * meters_per_deg_lat


def image_footprint_size_m(lat, zoom, gmap_scale, image_w, image_h):
    mpp = meters_per_pixel(lat, zoom, gmap_scale)
    width_m = image_w * mpp
    height_m = image_h * mpp
    return width_m, height_m


def projection_length_for_axis_aligned_rect(width_m, height_m, ux, uy):
    return abs(ux) * width_m + abs(uy) * height_m


def compute_line_overlap_from_centers(
    curr_center_m,
    prev_center_m,
    curr_w_m,
    curr_h_m,
    prev_w_m,
    prev_h_m,
    overlap_keep_ratio=1.0,
):
    dx = curr_center_m[0] - prev_center_m[0]
    dy = curr_center_m[1] - prev_center_m[1]
    center_dist_m = math.hypot(dx, dy)
    if center_dist_m <= 1e-9:
        ux, uy = 0.0, 1.0
        d = 0.0
    else:
        ux, uy = dx / center_dist_m, dy / center_dist_m
        d = center_dist_m

    curr_len = projection_length_for_axis_aligned_rect(curr_w_m, curr_h_m, ux, uy)
    prev_len = projection_length_for_axis_aligned_rect(prev_w_m, prev_h_m, ux, uy)
    if curr_len <= 0.0 or prev_len <= 0.0:
        return None

    a = curr_len / 2.0
    b = prev_len / 2.0
    low = max(-a, -d - b)
    high = min(a, -d + b)
    overlap_len_raw = max(0.0, high - low)
    keep_ratio = min(1.0, max(0.0, float(overlap_keep_ratio)))
    overlap_len = overlap_len_raw * keep_ratio
    overlap_ratio_curr = overlap_len / curr_len if curr_len > 0 else 0.0

    cut_side = "le"
    cut_t_m_current = low + overlap_len
    return {
        "ux": float(ux),
        "uy": float(uy),
        "center_distance_m": float(center_dist_m),
        "overlap_length_m": float(overlap_len),
        "overlap_ratio_current": float(overlap_ratio_curr),
        "cut_side": cut_side,
        "cut_t_m_current": float(cut_t_m_current),
    }


def point_projection_t_from_pixel(px, py, image_w, image_h, mpp, ux, uy):
    dx_m = (px - (image_w / 2.0)) * mpp
    dy_m = ((image_h / 2.0) - py) * mpp
    return (dx_m * ux) + (dy_m * uy)


def point_inside_line_overlap(px, py, image_w, image_h, mpp, ux, uy, cut_t_m, cut_side):
    t = point_projection_t_from_pixel(px, py, image_w, image_h, mpp, ux, uy)
    if cut_side == "le":
        return t <= cut_t_m
    return t >= cut_t_m


def tile_fully_inside_line_overlap(tile_x0, tile_y0, tile_x1, tile_y1, image_w, image_h, mpp, ux, uy, cut_t_m, cut_side):
    ts = [
        point_projection_t_from_pixel(tile_x0, tile_y0, image_w, image_h, mpp, ux, uy),
        point_projection_t_from_pixel(tile_x1, tile_y0, image_w, image_h, mpp, ux, uy),
        point_projection_t_from_pixel(tile_x1, tile_y1, image_w, image_h, mpp, ux, uy),
        point_projection_t_from_pixel(tile_x0, tile_y1, image_w, image_h, mpp, ux, uy),
    ]
    if cut_side == "le":
        return max(ts) <= cut_t_m
    return min(ts) >= cut_t_m


def get_line_cut_segment_px(image_w, image_h, mpp, ux, uy, cut_t_m):
    if mpp is None or mpp <= 0:
        return None
    cx = image_w / 2.0
    cy = image_h / 2.0
    n = np.array([ux, -uy], dtype=np.float64)
    c_px = cut_t_m / mpp
    p0 = np.array([cx, cy], dtype=np.float64) + (n * c_px)
    d = np.array([-n[1], n[0]], dtype=np.float64)
    span = float(max(image_w, image_h) * 2.0)
    p1 = (int(round(p0[0] - (d[0] * span))), int(round(p0[1] - (d[1] * span))))
    p2 = (int(round(p0[0] + (d[0] * span))), int(round(p0[1] + (d[1] * span))))
    ok, q1, q2 = cv2.clipLine((0, 0, int(image_w), int(image_h)), p1, p2)
    if not ok:
        return None
    return q1, q2


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


def process_image_yolo(image_path, yolo_model, args, metadata_map, cross_image_state=None):
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
    embedded_meta = extract_embedded_drone_metadata(image_path)
    line_lat = embedded_meta["lat"]
    line_lon = embedded_meta["lon"]

    embedded_w_m = embedded_meta.get("footprint_width_m")
    embedded_h_m = embedded_meta.get("footprint_height_m")
    has_embedded_footprint = (
        embedded_w_m is not None
        and embedded_h_m is not None
        and float(embedded_w_m) > 0.0
        and float(embedded_h_m) > 0.0
    )
    mpp_line = mpp
    if has_embedded_footprint:
        mpp_line = ((float(embedded_w_m) / float(w)) + (float(embedded_h_m) / float(h))) / 2.0

    curr_center_m = None
    curr_w_m = None
    curr_h_m = None
    line_overlap_info = None
    overlap_ratio_curr = None
    line_cut_segment_px = None

    if cross_image_state is not None and line_lat is not None and line_lon is not None and has_embedded_footprint:
        if cross_image_state.get("ref_lat") is None:
            cross_image_state["ref_lat"] = line_lat
        ref_lat = cross_image_state["ref_lat"]
        curr_center_m = image_center_global_m(line_lat, line_lon, ref_lat)
        curr_w_m = float(embedded_w_m)
        curr_h_m = float(embedded_h_m)

        prev_center_m = cross_image_state.get("prev_center_m")
        prev_w_m = cross_image_state.get("prev_width_m")
        prev_h_m = cross_image_state.get("prev_height_m")
        if (
            prev_center_m is not None
            and prev_w_m is not None
            and prev_h_m is not None
            and curr_w_m is not None
            and curr_h_m is not None
            and mpp_line is not None
        ):
            line_overlap_info = compute_line_overlap_from_centers(
                curr_center_m,
                prev_center_m,
                curr_w_m,
                curr_h_m,
                prev_w_m,
                prev_h_m,
                overlap_keep_ratio=args.line_overlap_keep_ratio,
            )
            if line_overlap_info is not None and line_overlap_info["overlap_ratio_current"] < args.min_overlap_ratio:
                line_overlap_info = None
            if line_overlap_info is not None and line_overlap_info["overlap_length_m"] > 0.0:
                overlap_ratio_curr = line_overlap_info["overlap_ratio_current"]
                line_cut_segment_px = get_line_cut_segment_px(
                    w,
                    h,
                    mpp_line,
                    line_overlap_info["ux"],
                    line_overlap_info["uy"],
                    line_overlap_info["cut_t_m_current"],
                )
    elif cross_image_state is not None and cross_image_state.get("prev_center_m") is not None:
        print(
            f"Warning: {filename} missing embedded metadata for overlap line; "
            "running full-image detection for this frame."
        )

    tile_dir = os.path.join(args.output_dir, f"{base}_tiles")
    if args.save_tile_preds:
        os.makedirs(tile_dir, exist_ok=True)

    raw_detections = []
    tiles_processed = 0
    tiles_skipped_in_overlap = 0
    boxes_skipped_in_overlap = 0
    names_map = {}
    for tile_id, x0, y0, x1, y1, tile_bgr in iterate_tiles(image_bgr, args.tile_size, args.tile_overlap):
        tiles_processed += 1
        if (
            line_overlap_info is not None
            and line_overlap_info["overlap_length_m"] > 0.0
            and mpp_line is not None
            and tile_fully_inside_line_overlap(
                x0,
                y0,
                x1,
                y1,
                w,
                h,
                mpp_line,
                line_overlap_info["ux"],
                line_overlap_info["uy"],
                line_overlap_info["cut_t_m_current"],
                line_overlap_info["cut_side"],
            )
        ):
            tiles_skipped_in_overlap += 1
            continue

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
            gcx = (gx1 + gx2) / 2.0
            gcy = (gy1 + gy2) / 2.0
            if (
                line_overlap_info is not None
                and line_overlap_info["overlap_length_m"] > 0.0
                and mpp_line is not None
                and point_inside_line_overlap(
                    gcx,
                    gcy,
                    w,
                    h,
                    mpp_line,
                    line_overlap_info["ux"],
                    line_overlap_info["uy"],
                    line_overlap_info["cut_t_m_current"],
                    line_overlap_info["cut_side"],
                )
            ):
                boxes_skipped_in_overlap += 1
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
    if cross_image_state is not None and curr_center_m is not None and curr_w_m is not None and curr_h_m is not None:
        cross_image_state["prev_center_m"] = curr_center_m
        cross_image_state["prev_width_m"] = curr_w_m
        cross_image_state["prev_height_m"] = curr_h_m

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
        file_obj.write(f"tiles_skipped_in_overlap: {tiles_skipped_in_overlap}\n")
        file_obj.write(f"boxes_skipped_in_overlap: {boxes_skipped_in_overlap}\n")
        file_obj.write(f"panel_count_boxes_raw: {len(raw_detections)}\n")
        file_obj.write(f"panel_count_boxes: {total_count}\n")
        file_obj.write(
            "overlap_ratio_of_current_image: "
            + (f"{overlap_ratio_curr:.6f}\n" if overlap_ratio_curr is not None else "None\n")
        )
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
    if line_cut_segment_px is not None:
        cv2.line(overlay, line_cut_segment_px[0], line_cut_segment_px[1], (0, 0, 139), 3)
    for row in box_rows:
        x1i, y1i, x2i, y2i = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        label_name = row["class_name"] if row["class_name"] is not None else "panel"
        label = f"{label_name} {row['confidence']:.2f}"
        cv2.putText(overlay, label, (x1i, max(0, y1i - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.output_dir, base + ".png"), overlay)

    print(
        f"Saved: {base}.png | tiles_processed={tiles_processed} | "
        f"overlap_tiles_skipped={tiles_skipped_in_overlap} | overlap_boxes_skipped={boxes_skipped_in_overlap} | "
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
    if os.path.splitext(args.model_path)[1].lower() != ".pt":
        raise ValueError("test.py is YOLO-only. Please provide a .pt model path.")

    metadata_map = load_metadata(args.metadata_csv)
    image_paths = collect_image_paths(args.input_dir)
    if not image_paths:
        print("No images found in", args.input_dir)
        return

    start = datetime.datetime.now()
    print("Backend: yolo")
    yolo_model = load_yolo_model(args.model_path)
    cross_image_state = {
        "ref_lat": None,
        "prev_center_m": None,
        "prev_width_m": None,
        "prev_height_m": None,
    }
    for path in image_paths:
        try:
            process_image_yolo(path, yolo_model, args, metadata_map=metadata_map, cross_image_state=cross_image_state)
        except Exception as exc:
            print(f"Error processing {path}: {exc}")

    print("Total time:", (datetime.datetime.now() - start))


if __name__ == "__main__":
    main()
