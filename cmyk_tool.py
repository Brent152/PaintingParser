import argparse
import os
import csv
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
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

def load_color_palette_for_viz(cmyk_data_file):
    """Load the color palette from the CMYK data CSV file for visualization."""
    colors = {}
    with open(cmyk_data_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = int(row['Index'])
            c, m, y, k = float(row['C']), float(row['M']), float(row['Y']), float(row['K'])
            # Convert CMYK to RGB
            r = int(255 * (1 - c) * (1 - k))
            g = int(255 * (1 - m) * (1 - k))
            b = int(255 * (1 - y) * (1 - k))
            colors[index] = {
                'rgb': (r, g, b),
                'name': row['Name'],
                'cmyk': (c, m, y, k),
                'count': int(row['Count'])
            }
    return colors

def load_index_map_for_viz(index_map_file):
    """Load the index map from CSV for visualization."""
    index_map = []
    with open(index_map_file, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            index_map.append([int(x) for x in row])
    return index_map

def create_simple_visualization(index_map, colors, cell_size=30, show_numbers=True):
    """Create a PNG visualization of the index map with actual colors."""
    height = len(index_map)
    width = len(index_map[0]) if index_map else 0
    img_width = width * cell_size
    img_height = height * cell_size
    
    # Create image
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", max(8, cell_size // 3))
    except:
        try:
            font = ImageFont.truetype("calibri.ttf", max(8, cell_size // 3))
        except:
            font = ImageFont.load_default()
    
    # Draw grid
    for y, row in enumerate(index_map):
        for x, color_index in enumerate(row):
            x_pos = x * cell_size
            y_pos = y * cell_size
            
            # Get color
            if color_index in colors:
                color = colors[color_index]['rgb']
            else:
                color = (128, 128, 128)  # Gray for missing colors
            
            # Draw cell
            draw.rectangle([x_pos, y_pos, x_pos + cell_size, y_pos + cell_size], 
                          fill=color, outline=(0, 0, 0))
            
            # Draw index number if enabled
            if show_numbers:
                text = str(color_index)
                # Calculate text color (black or white) based on background brightness
                brightness = sum(color) / 3
                text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
                
                # Center text in cell
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x_pos + (cell_size - text_width) // 2
                text_y = y_pos + (cell_size - text_height) // 2
                
                draw.text((text_x, text_y), text, fill=text_color, font=font)
    
    return img

def create_html_visualization_integrated(index_map_file, cmyk_data_file, output_file):
    """Create an HTML file with color visualization (integrated version)."""
    
    # Load colors
    colors = {}
    with open(cmyk_data_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = int(row['Index'])
            c, m, y, k = float(row['C']), float(row['M']), float(row['Y']), float(row['K'])
            r, g, b = cmyk_to_rgb(c, m, y, k)
            colors[index] = {
                'rgb': f'rgb({r},{g},{b})',
                'hex': f'#{r:02x}{g:02x}{b:02x}',
                'name': row['Name'],
                'cmyk': f'C:{c:.3f} M:{m:.3f} Y:{y:.3f} K:{k:.3f}',
                'count': row['Count']
            }
    
    # Load index map
    index_map = []
    with open(index_map_file, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            index_map.append([int(x) for x in row])
    
    # Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <title>CMYK Index Map Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .grid { display: inline-block; border: 2px solid black; }
        .cell { 
            width: 25px; height: 25px; 
            display: inline-block; 
            border: 1px solid #ccc;
            text-align: center;
            font-size: 10px;
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 1px black;
            line-height: 25px;
            cursor: pointer;
        }
        .legend {
            margin-top: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            background: #f0f0f0;
            padding: 5px;
            border-radius: 3px;
            margin: 2px;
        }
        .legend-color {
            width: 30px;
            height: 30px;
            border: 1px solid black;
            margin-right: 10px;
        }
        .legend-info {
            font-size: 12px;
        }
        .tooltip {
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 5px;
            border-radius: 3px;
            font-size: 11px;
            pointer-events: none;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <h1>CMYK Index Map Visualization</h1>
    <div class="grid">
"""
    
    # Generate grid
    for row in index_map:
        html += "        <div>\n"
        for color_index in row:
            color_data = colors.get(color_index, {'rgb': 'rgb(128,128,128)', 'name': 'Unknown', 'cmyk': 'Unknown', 'count': 0})
            html += f'            <div class="cell" style="background-color: {color_data["rgb"]}" title="Index {color_index}: {color_data["name"]} | {color_data["cmyk"]} | Count: {color_data["count"]}">{color_index}</div>\n'
        html += "        </div>\n"
    
    html += """    </div>
    
    <div class="legend">
        <h2>Color Palette</h2>
"""
    
    # Generate legend
    for index in sorted(colors.keys()):
        color_data = colors[index]
        html += f"""        <div class="legend-item">
            <div class="legend-color" style="background-color: {color_data['rgb']}"></div>
            <div class="legend-info">
                <strong>Index {index}:</strong> {color_data['name']}<br>
                {color_data['cmyk']}<br>
                Count: {color_data['count']}
            </div>
        </div>
"""
    
    html += """    </div>
</body>
</html>"""
    
    with open(output_file, 'w') as f:
        f.write(html)
    print(f"HTML visualization saved to: {output_file}")

def generate_visualization(image_path, index_map_file, cmyk_data_file, viz_type="simple"):
    """Generate visualization and clean up temporary files."""
    base, ext = os.path.splitext(image_path)
    
    if viz_type == "simple":
        # Create PNG visualization
        colors = load_color_palette_for_viz(cmyk_data_file)
        index_map = load_index_map_for_viz(index_map_file)
        
        print(f"Loaded {len(colors)} colors")
        print(f"Index map dimensions: {len(index_map)}x{len(index_map[0]) if index_map else 0}")
        
        # Create visualization
        img = create_simple_visualization(index_map, colors, cell_size=30, show_numbers=True)
        
        # Save visualization
        output_path = f"{base}_visualization.png"
        img.save(output_path)
        print(f"PNG visualization saved to: {output_path}")
        
    elif viz_type == "html":
        # Create HTML visualization
        output_path = f"{base}_visualization.html"
        create_html_visualization_integrated(index_map_file, cmyk_data_file, output_path)
    
    # Clean up temporary index map file
    try:
        os.remove(index_map_file)
        print(f"Deleted temporary index map file: {index_map_file}")
    except Exception as e:
        print(f"Could not delete {index_map_file}: {e}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Extract unique CMYK color counts from an image, with color reduction options."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--precision", type=float, default=0.05, help="Precision for merging similar colors (default: 0.05, only for precision method).")
    parser.add_argument("--save-cmyk", action="store_true", help="Also save the full CMYK pixel data.")
    parser.add_argument("--reconstruct", action="store_true", help="Reconstruct the image from the CMYK data (deletes CMYK data file if --save-cmyk is not set).")
    parser.add_argument("--output", type=str, default=None, help="Output image file name for reconstruction.")
    parser.add_argument("--scale", nargs='+', help="Scale the image: single float for percentage (e.g., 0.5 for 50%%) or two integers for WIDTH HEIGHT in pixels.")
    parser.add_argument("--resample", type=str, default="nearest", choices=["nearest", "bilinear", "bicubic", "lanczos"], help="Resampling filter for scaling (default: nearest)")
    parser.add_argument("--approx-method", type=str, default="precision", choices=["precision", "kmeans", "median_cut"], help="Color reduction method: precision, kmeans, or median_cut (default: precision)")
    parser.add_argument("--n-colors", type=int, default=16, help="Number of colors for kmeans or median_cut (default: 16)")
    parser.add_argument("--visualize", type=str, default="simple", choices=["simple", "html", "none"], help="Generate visualization: 'simple' for PNG (default), 'html' for interactive HTML, or 'none' to skip visualization")
    args = parser.parse_args()

    img = Image.open(args.image).convert('RGB')
    if args.scale:
        resample_dict = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }
        resample = resample_dict[args.resample]
        
        if len(args.scale) == 1:
            # Single value: treat as percentage
            try:
                scale_factor = float(args.scale[0])
                original_width, original_height = img.size
                width = int(original_width * scale_factor)
                height = int(original_height * scale_factor)
                print(f"Image scaled in memory by {scale_factor*100:.1f}% from {original_width}x{original_height} to {width}x{height} using {args.resample} resampling.")
            except ValueError:
                raise ValueError("Scale percentage must be a valid float (e.g., 0.5 for 50%)")
        elif len(args.scale) == 2:
            # Two values: treat as width and height in pixels
            try:
                width, height = int(args.scale[0]), int(args.scale[1])
                original_width, original_height = img.size
                print(f"Image scaled in memory from {original_width}x{original_height} to {width}x{height} using {args.resample} resampling.")
            except ValueError:
                raise ValueError("Scale dimensions must be valid integers for width and height")
        else:
            raise ValueError("Scale must be either 1 value (percentage) or 2 values (width height)")
        
        img = img.resize((width, height), resample)

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

    # Generate visualization if requested
    if args.visualize != "none":
        color_to_index = {tuple([round(x, 4) for x in color]): idx for idx, color in enumerate(color_list)}
        index_map_file = write_index_map(args.image, cmyk_data, width, height, color_to_index)
        # Generate the cmyk_data file path
        base, ext = os.path.splitext(args.image)
        cmyk_data_file = f"{base}_cmyk_data.csv"
        generate_visualization(args.image, index_map_file, cmyk_data_file, args.visualize)

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