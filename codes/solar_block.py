
import os
import math
import xml.etree.ElementTree as ET
from io import BytesIO
import datetime
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import matplotlib.pyplot as plt

# Your project imports (adjust python path if required)
from model_list import segnet_1, segnet_3, segnet_0, fast_scnn_2
import sys
sys.path.append("..")
from data_processing import prepare_data, data_processing_tool_4


def normalize_ring(latlon, precision=7):
    """
    Return a canonical, hashable representation of a closed ring (list of (lat,lon))
    that is invariant to rotation and reversal, and robust to tiny floating errors.
    Round coordinates to `precision` decimals.
    """
    # remove duplicated closing point if present
    pts = [(round(p[0], precision), round(p[1], precision)) for p in latlon]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]

    if len(pts) == 0:
        return ()

    n = len(pts)
    # generate rotations
    rotations = []
    for i in range(n):
        rot = tuple(pts[i:] + pts[:i])
        rotations.append(rot)
    # reversed rotations
    rpts = list(reversed(pts))
    for i in range(n):
        rot = tuple(rpts[i:] + rpts[:i])
        rotations.append(rot)

    # pick lexicographically smallest rotation as canonical representation
    canonical = min(rotations)
    return canonical


def read_kml_blocks(kml_path, debug=False):
    """
    Parse KML and return a list of unique blocks.
    Deduplicates identical polygons by canonicalizing rings.
    """
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(kml_path)
    root = tree.getroot()

    blocks = []
    seen = set()
    block_id = 1

    coords_nodes = root.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)

    if debug:
        print("Found coordinates nodes:", len(coords_nodes))

    for coord in coords_nodes:
        text = coord.text
        if not text:
            continue

        pts = []
        for token in text.strip().split():
            parts = token.split(",")
            if len(parts) < 2:
                continue
            lon = float(parts[0])
            lat = float(parts[1])
            pts.append((lat, lon))

        # require valid polygon
        if len(pts) < 3:
            continue

        key = normalize_ring(pts, precision=7)
        if key == ():
            continue

        if key in seen:
            if debug:
                print(f"Skipping duplicate polygon (normalized key present).")
            continue

        # mark seen and append block
        seen.add(key)

        blocks.append({
            "id": block_id,
            "name": f"block-{block_id}",
            "polygon": pts,
            "area_m2": polygon_area_m2(pts),
            "centroid": polygon_centroid(pts)
        })
        block_id += 1

    if debug:
        print(f"Unique blocks parsed: {len(blocks)}")

    return blocks



def polygon_centroid(latlon):
    """
    Centroid (approx) of polygon in lat/lon using arithmetic mean of vertices.
    For small polygons this is sufficient.
    """
    lat = np.mean([p[0] for p in latlon])
    lon = np.mean([p[1] for p in latlon])
    return (lat, lon)


def polygon_area_m2(latlon):
    """
    Approx polygon area in square meters via planar projection at polygon mean latitude.
    Good for small areas (solar farms).
    """
    lat0 = np.mean([p[0] for p in latlon])
    m_per_deg_lat = 111320.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat0))
    xy = np.array([ (p[1] * m_per_deg_lon, p[0] * m_per_deg_lat) for p in latlon ])
    x = xy[:,0]; y = xy[:,1]
    area = 0.5 * abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))
    return float(area)


# -------------------------
# Google Static Maps helpers
# -------------------------
def meters_per_pixel(lat_deg, zoom, scale):
    """Google Web Mercator ground resolution (m/px) adjusted by scale."""
    lat_rad = math.radians(lat_deg)
    return (156543.03392 * math.cos(lat_rad) / (2 ** zoom)) / scale


def tile_top_left_bottom_right(center_lat, center_lon, zoom, scale, image_px):
    """Return top-left (lat,lon) and bottom-right (lat,lon) for a tile centered at center_lat,center_lon."""
    mpp = meters_per_pixel(center_lat, zoom, scale)
    half_m = (image_px / 2.0) * mpp
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(center_lat))
    dlat = half_m / meters_per_deg_lat
    dlon = half_m / meters_per_deg_lon
    top_left = (center_lat + dlat, center_lon - dlon)
    bottom_right = (center_lat - dlat, center_lon + dlon)
    return top_left, bottom_right


def download_static_map(center_lat, center_lon, zoom, size_px, scale, api_key, maptype="satellite"):
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={center_lat},{center_lon}"
        f"&zoom={zoom}"
        f"&size={size_px}x{size_px}"
        f"&scale={scale}"
        f"&maptype={maptype}"
        f"&key={api_key}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


# -------------------------
# Geo -> pixel projection for a tile
# -------------------------
def polygon_latlon_to_tile_pixels(polygon_latlon, center_lat, center_lon, zoom, scale, image_px):
    """
    Project polygon lat/lon to pixel coords inside a tile (0..image_px-1).
    Center maps to (image_px/2, image_px/2). Returns list of (x,y) floats.
    """
    mpp = meters_per_pixel(center_lat, zoom, scale)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(center_lat))
    pixel_coords = []
    for lat, lon in polygon_latlon:
        dx_m = (lon - center_lon) * meters_per_deg_lon
        dy_m = (lat - center_lat) * meters_per_deg_lat
        px = (image_px / 2.0) + (dx_m / mpp)
        py = (image_px / 2.0) - (dy_m / mpp)   # increase y downwards
        pixel_coords.append((px, py))
    return pixel_coords


# -------------------------
# Model loading & prediction helpers
# -------------------------
def load_model_for_type(model_type, model_name, input_shape):
    if model_type == 1:
        model = fast_scnn_2.fast_scnn_v2(input_shape=input_shape, batch_size=1, n_labels=2, model_summary=False)
    elif model_type == 2:
        model = segnet_3.segnet_resnet_v2(input_shape=input_shape, batch_size=1, n_labels=2, model_summary=False)
    elif model_type == 3:
        model = segnet_1.segnet_4_encoder_decoder(input_shape=input_shape, batch_size=1, n_labels=2, model_summary=False)
    else:
        model = segnet_0.segnet_original(input_shape=input_shape, batch_size=1, n_labels=2, model_summary=False)

    model.load_weights(os.path.join(MODEL_PATH, model_name))
    return model


def predict_tile_mask(tile_arr, model, threshold=0.2):
    """
    tile_arr: numpy RGB array (H,W,3)
    returns: binary mask (H, W) where True indicates predicted solar panel
    """
    sub_imgs, padded_img, padded_w, padded_h = data_processing_tool_4.get_sub_images(tile_arr)
    # Predict
    preds = get_predicted_label_list(sub_imgs, model, threshold=threshold)
    full_label = data_processing_tool_4.get_full_predicted_label(padded_h, padded_w, preds)
    if isinstance(full_label, Image.Image):
        full_arr = np.asarray(full_label)
    else:
        full_arr = np.asarray(full_label)
    # convert RGB->binary
    if full_arr.ndim == 3:
        rgb = full_arr[:, :, :3]
        pixels = rgb.reshape(-1, 3)
        uniq_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        bg_color = uniq_colors[counts.argmax()]
        mask = np.any(rgb != bg_color.reshape(1,1,3), axis=2)
    else:
        mask = full_arr > 0
    # crop/resize mask to original tile size if needed
    mask_h, mask_w = mask.shape
    if (mask_h, mask_w) != (tile_arr.shape[0], tile_arr.shape[1]):
        mask_pil = Image.fromarray((mask.astype('uint8')*255))
        mask_resized = mask_pil.resize((tile_arr.shape[1], tile_arr.shape[0]), resample=Image.NEAREST)
        mask = np.array(mask_resized).astype(bool)
    return mask, full_label, padded_img


# reuse earlier get_predicted_label_list
def get_predicted_label_list(sub_imgs, model, threshold=0.2):
    sub_predicted_label_list = []
    # try to use INITIAL_BATCH_SIZE if defined globally
    batch_size_global = INITIAL_BATCH_SIZE if 'INITIAL_BATCH_SIZE' in globals() else 8
    total_batches = math.ceil(len(sub_imgs) / batch_size_global)
    for i in range(total_batches):
        start = i * batch_size_global
        end = start + batch_size_global
        batch = sub_imgs[start:end]
        results = model.predict(batch)
        for r in results:
            prob_map = r
            custom_result = np.zeros_like(prob_map)
            object_mask = prob_map[:, :, 1] > threshold
            custom_result[object_mask, 1] = 1
            custom_result[~object_mask, 0] = 1
            my_img = prepare_data.onehot_to_rgb(custom_result, prepare_data.id2code)
            sub_predicted_label_list.append(my_img)
    return sub_predicted_label_list


# -------------------------
# Color utilities
# -------------------------
BLOCK_COLORS = [
    (0, 0, 255, 120),    # blue
    (0, 255, 0, 120),    # green
    (255, 255, 0, 120),  # yellow
    (148, 0, 211, 120),  # violet
    (255, 165, 0, 120),  # orange
    (0, 255, 255, 120),  # cyan
    (255, 0, 255, 120),  # magenta
    (128, 128, 0, 120),  # olive
]


# -------------------------
# Main orchestrator
# -------------------------
def run_overlap_tile_for_all_blocks(kml_path,
                                    api_key,
                                    zoom=19,
                                    image_px=1024,
                                    scale=2,
                                    maptype="satellite",
                                    model_type=1,
                                    model_name="fast_scnn_2.h5",
                                    panel_area_m2=12.5,
                                    out_dir="predict-gmap-2",
                                    input_tile_dir="save_lat_lon",
                                    threshold=0.2):
    # 1. Read blocks
    blocks = read_kml_blocks("kml/lastest_all.kml", debug=True) ##hard-coded kml input.
    if not blocks:
        raise RuntimeError("No block polygons found in KML.")

    # 2. Centroids and average centroid
    centroids = [b['centroid'] for b in blocks]
    avg_lat = float(np.mean([c[0] for c in centroids]))
    avg_lon = float(np.mean([c[1] for c in centroids]))

    # 3. Download tile centered at average centroid
    tile_img = download_static_map(avg_lat, avg_lon, zoom, image_px, scale, api_key, maptype)
    upsample = 4 # try 2 or 4 (Valid with 2)
    orig_image_px = image_px
    if upsample > 1:
        new_px = image_px * upsample
        tile_img = tile_img.resize((new_px, new_px), resample=Image.BICUBIC)
        tile_arr = np.asarray(tile_img)
        image_px = new_px
    # tile_arr = np.asarray(tile_img)
    if tile_arr.ndim == 3 and tile_arr.shape[2] == 4:
        tile_arr = tile_arr[:, :, :3]

    # 4. Load model (need input shape)
    sub_imgs_dummy, padded_dummy, pw_dummy, ph_dummy = data_processing_tool_4.get_sub_images(tile_arr)
    model = load_model_for_type(model_type, model_name, input_shape=sub_imgs_dummy[0].shape)

    # 5. Predict mask and get full_label / padded_img (for final overlay)
    pred_mask, full_label, padded_img = predict_tile_mask(tile_arr, model, threshold=threshold)

    # 6. For each block, rasterize polygon into tile pixels and compute overlap
    block_results = []
    for i, block in enumerate(blocks):
        # pix_poly = polygon_latlon_to_tile_pixels(block['polygon'], avg_lat, avg_lon, zoom, scale, image_px)
        
        # project in ORIGINAL resolution
        pix_poly_orig = polygon_latlon_to_tile_pixels(
            block['polygon'],
            avg_lat, avg_lon,
            zoom, scale,
            orig_image_px
        )

        # scale polygon pixels to upsampled image
        pix_poly = [
            (x * upsample, y * upsample)
            for x, y in pix_poly_orig
        ]

        # Clip polygon coordinates and convert to tuple list
        pix_poly_clipped = [(max(0, min(image_px-1, x)), max(0, min(image_px-1, y))) for x,y in pix_poly]

        # Make mask
        roi_mask_img = Image.new("L", (image_px, image_px), 0)
        draw = ImageDraw.Draw(roi_mask_img)
        # If polygon is degenerate (all vertices outside), ImageDraw still handles it.
        draw.polygon(pix_poly_clipped, outline=1, fill=1)
        roi_mask = np.array(roi_mask_img).astype(bool)

        # Intersection with predicted mask
        overlap_mask = pred_mask & roi_mask
        overlap_pixels = int(overlap_mask.sum())
        # mpp = meters_per_pixel(avg_lat, zoom, scale)
        # overlap_area_m2 = overlap_pixels * (mpp ** 2)
        
        mpp = meters_per_pixel(avg_lat, zoom, scale)
        effective_mpp = mpp / upsample
        overlap_area_m2 = overlap_pixels * (effective_mpp ** 2)

        est_panels_float = overlap_area_m2 / panel_area_m2 if panel_area_m2 > 0 else 0.0
        est_panels_int = int(math.floor(est_panels_float))

        block_results.append({
            "id": block['id'],
            "name": block['name'],
            "polygon": block['polygon'],
            "polygon_pixels": pix_poly_clipped,
            "block_area_m2": block['area_m2'],
            "overlap_pixels": overlap_pixels,
            "overlap_area_m2": overlap_area_m2,
            "est_panels_float": est_panels_float,
            "est_panels_int": est_panels_int
        })

    # 7. Build overlay image:
    #    - base is tile_img (RGB)
    #    - overlay colored block fills (semi-transparent)
    #    - predicted solar mask in red (solid-ish)
    #    - block outlines
    base_rgb = tile_img.convert("RGBA")
    overlay = Image.new("RGBA", base_rgb.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # draw blocks with colors
    for idx, br in enumerate(block_results):
        color = BLOCK_COLORS[idx % len(BLOCK_COLORS)]
        # polygon may have fractional coords; cast to tuples
        poly_xy = [ (float(x), float(y)) for x,y in br['polygon_pixels'] ]
        draw.polygon(poly_xy, fill=color)
        # outline: darker variant (use RGB with full alpha)
        outline_rgb = (max(0,color[0]-30), max(0,color[1]-30), max(0,color[2]-30), 255)
        draw.line(poly_xy + [poly_xy[0]], fill=outline_rgb, width=2)

    # draw predicted solar mask in red
    red_mask_img = Image.new("RGBA", base_rgb.size, (255,0,0,0))
    red_draw = ImageDraw.Draw(red_mask_img, "RGBA")
    # convert pred_mask to polygon(s) by drawing pixels (fast approach: create image from mask)
    pred_mask_img = Image.fromarray((pred_mask.astype('uint8')*255)).convert("L")
    pred_rgba = Image.new("RGBA", base_rgb.size, (255,0,0,140))  # semi-transparent red
    # composite only where pred_mask_img > 0
    overlay = Image.alpha_composite(overlay, Image.composite(pred_rgba, Image.new("RGBA", base_rgb.size, (0,0,0,0)), pred_mask_img))

    # combine base + overlay
    final = Image.alpha_composite(base_rgb, overlay)

    # add legend / annotations (optional)
    try:
        fnt = ImageFont.load_default()
    except Exception:
        fnt = None
    legend_draw = ImageDraw.Draw(final)
    legend_x = 10
    legend_y = 10
    legend_draw.rectangle([legend_x-4, legend_y-4, legend_x+180, legend_y+ (16 * (len(block_results)+2))], fill=(255,255,255,200))
    legend_draw.text((legend_x, legend_y), f"Tile center (avg): {avg_lat:.6f}, {avg_lon:.6f}", fill=(0,0,0), font=fnt)
    ly = legend_y + 16
    for idx, br in enumerate(block_results):
        c = BLOCK_COLORS[idx % len(BLOCK_COLORS)]
        legend_draw.rectangle([legend_x, ly, legend_x+12, ly+12], fill=c)
        legend_draw.text((legend_x+16, ly), f"{br['name']}: overlap m2={br['overlap_area_m2']:.2f}, panels={br['est_panels_int']}", fill=(0,0,0), font=fnt)
        ly += 16
    # show red marker for solar mask count
    legend_draw.rectangle([legend_x, ly, legend_x+12, ly+12], fill=(255,0,0,160))
    legend_draw.text((legend_x+16, ly), f"Predicted solar mask (red)", fill=(0,0,0), font=fnt)

    # 8. Prepare filenames
    blocks_str = "_".join([str(b['id']) for b in blocks])
    safe_lat = f"{avg_lat:.6f}"
    safe_lon = f"{avg_lon:.6f}"
    base_name = f"kmz-zm-{zoom}_{safe_lat}_{safe_lon}_blocks-{blocks_str}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(input_tile_dir, exist_ok=True)
    tile_path = os.path.join(input_tile_dir, base_name + ".png")
    final_path = os.path.join(out_dir, base_name + ".png")
    txt_path = os.path.join(out_dir, base_name + ".txt")

    # Save the raw downloaded tile (input)
    tile_img.save(tile_path, "PNG")
    # Save final overlay
    final.convert("RGB").save(final_path, "PNG")

    # 9. Compose txt file
    top_left, bottom_right = tile_top_left_bottom_right(avg_lat, avg_lon, zoom, scale, image_px)
    # total_mask_pixels = int(pred_mask.sum())
    # total_mask_area_m2 = total_mask_pixels * (meters_per_pixel(avg_lat, zoom, scale) ** 2)
    total_mask_pixels = int(pred_mask.sum())
    mpp = meters_per_pixel(avg_lat, zoom, scale)
    effective_mpp = mpp / upsample
    total_mask_area_m2 = total_mask_pixels * (effective_mpp ** 2)

    with open(txt_path, "w") as fh:
        fh.write(f"input_tile: {os.path.basename(tile_path)}\n")
        fh.write(f"tile_center_avg_lat: {avg_lat:.8f}\n")
        fh.write(f"tile_center_avg_lon: {avg_lon:.8f}\n")
        fh.write(f"top_left_lat: {top_left[0]:.8f}\n")
        fh.write(f"top_left_lon: {top_left[1]:.8f}\n")
        fh.write(f"bottom_right_lat: {bottom_right[0]:.8f}\n")
        fh.write(f"bottom_right_lon: {bottom_right[1]:.8f}\n")
        fh.write(f"zoom: {zoom}\n")
        fh.write(f"scale: {scale}\n")
        fh.write(f"image_px: {image_px}\n")
        fh.write(f"meters_per_pixel: {meters_per_pixel(avg_lat, zoom, scale):.6f}\n")
        fh.write(f"total_predicted_mask_pixels: {total_mask_pixels}\n")
        fh.write(f"total_predicted_mask_area_m2: {total_mask_area_m2:.4f}\n")
        fh.write(f"panel_area_m2_input: {panel_area_m2:.4f}\n")
        fh.write("\nblocks:\n")
        for br in block_results:
            fh.write(f"  - id: {br['id']}\n")
            fh.write(f"    name: {br['name']}\n")
            fh.write(f"    block_area_m2: {br['block_area_m2']:.4f}\n")
            fh.write(f"    vertices:\n")
            for vlat, vlon in br['polygon']:
                fh.write(f"      - {vlat:.8f}, {vlon:.8f}\n")
            fh.write(f"    overlap_pixels: {br['overlap_pixels']}\n")
            fh.write(f"    overlap_area_m2: {br['overlap_area_m2']:.4f}\n")
            fh.write(f"    est_panels_float: {br['est_panels_float']:.6f}\n")
            fh.write(f"    est_panels_int_floor: {br['est_panels_int']}\n")
            fh.write("\n")
        fh.write(f"calculation_timestamp_utc: {datetime.datetime.utcnow().isoformat()}Z\n")

    # 10. Append to a global summary file in out_dir
    summary_file = os.path.join(out_dir, "areas_summary.txt")
    with open(summary_file, "a") as sf:
        line = f"{base_name}\t{total_mask_pixels}\t{total_mask_area_m2:.4f}\t" + \
               ",".join([f"{b['id']}:{b['overlap_pixels']}:{b['overlap_area_m2']:.2f}:{b['est_panels_int']}" for b in block_results]) + "\n"
        sf.write(line)

    print(f"Saved overlay: {final_path}")
    print(f"Saved metadata: {txt_path}")
    return final_path, txt_path


# -------------------------
# MAIN (hard-coded configuration)
# -------------------------
def main():
    # ----- configure here -----
    KML_PATH = "kml/lastest_all.kml"                      # input KML with polygons (blocks)   
    ZOOM = 18 # 19
    IMAGE_PX = 1280
    SCALE = 2
    MAPTYPE = "satellite"
    MODEL_TYPE = 1
    MODEL_NAME = "fast_scnn_2.h5"
    PANEL_AREA_M2 = 2 ##(2.57, 101 panels) (2.2, 118)
    OUT_DIR = "predict-u4"
    INPUT_TILE_DIR = "save_u4"
    THRESHOLD = 0.2
    # ---------------------------

    # Make sure model/path globals are available for helper functions
    global MODEL_PATH, INITIAL_BATCH_SIZE
    MODEL_PATH = "trained_models"
    INITIAL_BATCH_SIZE = 4

    # Run
    run_overlap_tile_for_all_blocks(
        kml_path=KML_PATH,
        api_key=API_KEY,
        API_KEY = "Use your own API key" ,
        zoom=ZOOM,
        image_px=IMAGE_PX,
        scale=SCALE,
        maptype=MAPTYPE,
        model_type=MODEL_TYPE,
        model_name=MODEL_NAME,
        panel_area_m2=PANEL_AREA_M2,
        out_dir=OUT_DIR,
        input_tile_dir=INPUT_TILE_DIR,
        threshold=THRESHOLD
    )


if __name__ == "__main__":
    main()
