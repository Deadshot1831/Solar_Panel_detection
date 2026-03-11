import argparse
import csv
import os
from PIL import Image

VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def split_image_into_windows(image_path, output_dir, window_width, window_height, pad_edges=True):
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    metadata_path = os.path.join(output_dir, f"{base_name}_tiles.csv")

    tile_count = 0
    with open(metadata_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "tile_file",
                "x_start",
                "y_start",
                "x_end",
                "y_end",
                "tile_width",
                "tile_height",
                "padded",
            ]
        )

        for y in range(0, img_height, window_height):
            for x in range(0, img_width, window_width):
                x_end = min(x + window_width, img_width)
                y_end = min(y + window_height, img_height)
                tile = image.crop((x, y, x_end, y_end))

                padded = False
                if pad_edges and tile.size != (window_width, window_height):
                    padded_tile = Image.new("RGB", (window_width, window_height), (0, 0, 0))
                    padded_tile.paste(tile, (0, 0))
                    tile = padded_tile
                    padded = True

                tile_name = f"{base_name}_x{x}_y{y}.png"
                tile_path = os.path.join(output_dir, tile_name)
                tile.save(tile_path)

                writer.writerow(
                    [
                        tile_name,
                        x,
                        y,
                        x_end,
                        y_end,
                        tile.width,
                        tile.height,
                        int(padded),
                    ]
                )
                tile_count += 1

    return tile_count, metadata_path, img_width, img_height


def collect_images(input_dir):
    images = []
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for name in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, name)
        if os.path.isfile(full_path) and name.lower().endswith(VALID_IMAGE_EXTENSIONS):
            images.append(full_path)
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Split drone image into small user-defined pixel windows."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to one input image. If omitted, all images from --input-dir are processed.",
    )
    parser.add_argument(
        "--input-dir",
        default="drone_imgs",
        help="Directory containing images to process when --image is not provided.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        required=True,
        help="Window width in pixels (example: 256)",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=None,
        help="Window height in pixels (default: same as window-width)",
    )
    parser.add_argument(
        "--output-dir",
        default="drone_img_windows",
        help="Directory to save window images and tile CSV",
    )
    parser.add_argument(
        "--no-pad-edges",
        action="store_true",
        help="Do not pad edge windows; edge tiles may be smaller than window size",
    )
    args = parser.parse_args()

    if args.window_width <= 0:
        raise ValueError("--window-width must be > 0")

    window_height = args.window_height if args.window_height is not None else args.window_width
    if window_height <= 0:
        raise ValueError("--window-height must be > 0")

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = collect_images(args.input_dir)
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {args.input_dir} with extensions: {', '.join(VALID_IMAGE_EXTENSIONS)}"
            )

    total_tiles = 0
    print(f"Window size: {args.window_width}x{window_height}")
    print(f"Output directory: {args.output_dir}")

    for image_path in image_paths:
        tile_count, metadata_path, img_width, img_height = split_image_into_windows(
            image_path=image_path,
            output_dir=args.output_dir,
            window_width=args.window_width,
            window_height=window_height,
            pad_edges=not args.no_pad_edges,
        )
        total_tiles += tile_count
        print(f"Processed: {image_path} | size={img_width}x{img_height} | tiles={tile_count}")
        print(f"Tile metadata CSV: {metadata_path}")

    print(f"Images processed: {len(image_paths)}")
    print(f"Total tiles generated: {total_tiles}")


if __name__ == "__main__":
    main()
