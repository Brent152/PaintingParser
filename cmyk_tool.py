import argparse
import os
import csv
from collections import Counter
from PIL import Image
import webcolors
import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

def rgb_to_cmyk(r, g, b):
    r, g, b = [x / 255.0 for x in (r, g, b)]
    k = 1 - max(r, g, b)
    if k == 1:
        return 0, 0, 0, 1
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return c, m, y, k

def cmyk_to_rgb(c, m, y, k):
    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)
    return int(round(r)), int(round(g)), int(round(b))

def closest_color_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(rgb_tuple, spec='css3')
    except ValueError:
        min_colors = {}
        for hex_value, name in webcolors._definitions._get_hex_to_name_map("css3").items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
            rd = (r_c - rgb_tuple[0]) ** 2
            gd = (g_c - rgb_tuple[1]) ** 2
            bd = (b_c - rgb_tuple[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        closest_name = min_colors[min(min_colors.keys())]
        return closest_name

def quantize_cmyk(cmyk, precision=0.05):
    return tuple(round(channel / precision) * precision for channel in cmyk)

def get_cmyk_pixels_and_counts(img, precision=0.05):
    width, height = img.size
    cmyk_pixels = []
    color_counter = Counter()
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            cmyk = rgb_to_cmyk(r, g, b)
            quant_cmyk = quantize_cmyk(cmyk, precision)
            row.extend([round(x, 4) for x in quant_cmyk])
            color_counter[quant_cmyk] += 1
        cmyk_pixels.append(row)
    return cmyk_pixels, color_counter, width, height

def get_cmyk_pixels_and_counts_from_rgb_array(rgb_array):
    height, width, _ = rgb_array.shape
    cmyk_pixels = []
    color_counter = Counter()
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = rgb_array[y, x]
            cmyk = rgb_to_cmyk(r, g, b)
            row.extend([round(x, 4) for x in cmyk])
            color_counter[cmyk] += 1
        cmyk_pixels.append(row)
    return cmyk_pixels, color_counter, width, height

def write_cmyk_data(image_path, color_counter):
    base, ext = os.path.splitext(image_path)
    counts_file = f"{base}_cmyk_data.csv"
    color_list = list(color_counter.keys())
    with open(counts_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Name", "C", "M", "Y", "K", "Count"])
        for idx, color in enumerate(color_list):
            rgb = cmyk_to_rgb(*color)
            color_name = closest_color_name(rgb)
            count = color_counter[color]
            writer.writerow([idx, color_name] + [round(x, 4) for x in color] + [count])
    print(f"Unique color counts written to {counts_file}")
    return color_list  # Return the color list for index mapping

def write_cmyk_pixels(image_path, cmyk_data, width, height):
    base, ext = os.path.splitext(image_path)
    cmyk_file = f"{base}_cmyk.csv"
    with open(cmyk_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([width, height])
        writer.writerows(cmyk_data)
    print(f"CMYK values written to {cmyk_file}")
    return cmyk_file

def write_index_map(image_path, cmyk_data, width, height, color_to_index):
    base, ext = os.path.splitext(image_path)
    indexed_map_file = f"{base}_cmyk_index_map.csv"
    with open(indexed_map_file, "w", newline="") as f:
        writer = csv.writer(f)
        for row in cmyk_data:
            index_row = []
            for i in range(0, len(row), 4):
                cmyk = tuple(row[i:i+4])
                idx = color_to_index[cmyk]
                index_row.append(idx)
            writer.writerow(index_row)
    print(f"Indexed CMYK map written to {indexed_map_file}")
    return indexed_map_file

def read_cmyk_csv(cmyk_file):
    with open(cmyk_file, "r", newline="") as f:
        reader = csv.reader(f)
        dims = next(reader)
        width, height = int(dims[0]), int(dims[1])
        cmyk_rows = []
        for row in reader:
            cmyk_row = []
            for i in range(0, len(row), 4):
                c, m, y, k = map(float, row[i:i+4])
                cmyk_row.append((c, m, y, k))
            cmyk_rows.append(cmyk_row)
    return cmyk_rows, width, height

def reconstruct_image_from_cmyk(cmyk_file, output_path=None):
    cmyk_rows, width, height = read_cmyk_csv(cmyk_file)
    img = Image.new("RGB", (width, height))
    for y, row in enumerate(cmyk_rows):
        for x, cmyk in enumerate(row):
            rgb = cmyk_to_rgb(*cmyk)
            img.putpixel((x, y), rgb)
    if output_path is None:
        base = cmyk_file.rsplit("_cmyk.csv", 1)[0]
        output_path = f"{base}_reconstructed.png"
    img.save(output_path)
    print(f"Image saved as {output_path}")

def quantize_image_kmeans(img, n_colors=16):
    if KMeans is None:
        raise ImportError("scikit-learn is required for kmeans quantization.")
    arr = np.array(img)
    shape = arr.shape
    arr = arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init=4)
    labels = kmeans.fit_predict(arr)
    new_colors = np.round(kmeans.cluster_centers_).astype('uint8')
    quantized_arr = new_colors[labels].reshape(shape)
    quantized_img = Image.fromarray(quantized_arr)
    return quantized_img

def main():
    parser = argparse.ArgumentParser(
        description="Extract unique CMYK color counts from an image, with color reduction options."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--precision", type=float, default=0.05, help="Precision for merging similar colors (default: 0.05, only for precision method).")
    parser.add_argument("--save-cmyk", action="store_true", help="Also save the full CMYK pixel data.")
    parser.add_argument("--reconstruct", action="store_true", help="Reconstruct the image from the CMYK data (deletes CMYK data file if --save-cmyk is not set).")
    parser.add_argument("--output", type=str, default=None, help="Output image file name for reconstruction.")
    parser.add_argument("--scale", nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help="Scale the image to WIDTH HEIGHT in memory before processing.")
    parser.add_argument("--resample", type=str, default="nearest", choices=["nearest", "bilinear", "bicubic", "lanczos"], help="Resampling filter for scaling (default: nearest)")
    parser.add_argument("--approx-method", type=str, default="precision", choices=["precision", "kmeans", "median_cut"], help="Color reduction method: precision, kmeans, or median_cut (default: precision)")
    parser.add_argument("--n-colors", type=int, default=16, help="Number of colors for kmeans or median_cut (default: 16)")
    parser.add_argument("--index-map", action="store_true", help="Output an indexed CMYK map CSV with color indices for each pixel, formatted as the image.")
    args = parser.parse_args()

    img = Image.open(args.image).convert('RGB')
    if args.scale:
        width, height = args.scale
        resample_dict = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }
        resample = resample_dict[args.resample]
        img = img.resize((width, height), resample)
        print(f"Image scaled in memory to {width}x{height} using {args.resample} resampling.")

    # Color reduction
    if args.approx_method == "precision":
        cmyk_data, color_counter, width, height = get_cmyk_pixels_and_counts(img, args.precision)
    elif args.approx_method == "kmeans":
        quantized_img = quantize_image_kmeans(img, n_colors=args.n_colors)
        rgb_array = np.array(quantized_img)
        cmyk_data, color_counter, width, height = get_cmyk_pixels_and_counts_from_rgb_array(rgb_array)
    elif args.approx_method == "median_cut":
        quantized_img = img.quantize(colors=args.n_colors, method=0).convert("RGB")
        rgb_array = np.array(quantized_img)
        cmyk_data, color_counter, width, height = get_cmyk_pixels_and_counts_from_rgb_array(rgb_array)
    else:
        raise ValueError("Unknown approximation method.")

    print(f"Number of unique colors: {len(color_counter)}")
    color_list = write_cmyk_data(args.image, color_counter)

    # Indexed CMYK map output (image-shaped index map)
    if args.index_map:
        color_to_index = {tuple([round(x, 4) for x in color]): idx for idx, color in enumerate(color_list)}
        write_index_map(args.image, cmyk_data, width, height, color_to_index)

    cmyk_file = None
    if args.save_cmyk or args.reconstruct:
        cmyk_file = write_cmyk_pixels(args.image, cmyk_data, width, height)

    if args.reconstruct:
        if not cmyk_file:
            print("Error: --reconstruct requires the CMYK data file to exist.")
            return
        reconstruct_image_from_cmyk(cmyk_file, args.output)
        if not args.save_cmyk:
            try:
                os.remove(cmyk_file)
                print(f"Deleted temporary CMYK data file: {cmyk_file}")
            except Exception as e:
                print(f"Could not delete {cmyk_file}: {e}")

if __name__ == "__main__":
    main()

# Pip Intalls:
# pip install pillow numpy scikit-learn webcolors==24.11.1

# Example usage:
# python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 16 32 --approx-method kmeans --n-colors 32 --reconstruct --index-map
# python cmyk_tool.py --image .\SkullsAndRoses\SkullsAndRoses.png --scale 16 32 --approx-method kmeans --n-colors 32 --reconstruct --index-map
# python cmyk_tool.py --image .\sam\sam.png --scale 512 512 --approx-method kmeans --n-colors 32 --reconstruct --index-map