import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os

def load_color_palette(cmyk_data_file):
    """Load the color palette from the CMYK data CSV file."""
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

def load_index_map(index_map_file):
    """Load the index map from CSV."""
    index_map = []
    with open(index_map_file, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            index_map.append([int(x) for x in row])
    return np.array(index_map)

def create_color_visualization(index_map, colors, cell_size=20, show_numbers=True):
    """Create a visual representation of the index map with actual colors."""
    height, width = index_map.shape
    img_width = width * cell_size
    img_height = height * cell_size
    
    # Create image
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", max(8, cell_size // 3))
    except:
        font = ImageFont.load_default()
    
    for y in range(height):
        for x in range(width):
            color_index = index_map[y, x]
            
            # Get color
            if color_index in colors:
                color = colors[color_index]['rgb']
            else:
                color = (255, 0, 255)  # Magenta for missing colors
            
            # Draw rectangle
            x1, y1 = x * cell_size, y * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=1)
            
            # Add index number
            if show_numbers and cell_size >= 15:
                text = str(color_index)
                # Calculate text color (black or white based on background)
                brightness = sum(color) / 3
                text_color = 'white' if brightness < 128 else 'black'
                
                # Center text in cell
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x1 + (cell_size - text_width) // 2
                text_y = y1 + (cell_size - text_height) // 2
                
                draw.text((text_x, text_y), text, fill=text_color, font=font)
    
    return img

def create_color_palette_legend(colors, cell_size=30, cols=8):
    """Create a color palette legend showing index, color, and name."""
    color_count = len(colors)
    rows = (color_count + cols - 1) // cols
    
    # Calculate dimensions
    legend_width = cols * cell_size * 4  # Extra space for text
    legend_height = rows * cell_size
    
    img = Image.new('RGB', (legend_width, legend_height), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, (index, color_data) in enumerate(sorted(colors.items())):
        row = i // cols
        col = i % cols
        
        # Color square
        x1 = col * cell_size * 4
        y1 = row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        
        draw.rectangle([x1, y1, x2, y2], fill=color_data['rgb'], outline='black')
        
        # Text
        text = f"{index}: {color_data['name']} ({color_data['count']})"
        draw.text((x2 + 5, y1 + 5), text, fill='black', font=font)
    
    return img

def main():
    parser = argparse.ArgumentParser(description="Visualize CMYK index map with actual colors")
    parser.add_argument("--index-map", required=True, help="Path to the index map CSV file")
    parser.add_argument("--cmyk-data", required=True, help="Path to the CMYK data CSV file")
    parser.add_argument("--cell-size", type=int, default=30, help="Size of each cell in pixels (default: 30)")
    parser.add_argument("--output", help="Output image path (default: based on input filename)")
    parser.add_argument("--no-numbers", action="store_true", help="Don't show index numbers on cells")
    parser.add_argument("--legend", action="store_true", help="Also create a color palette legend")
    
    args = parser.parse_args()
    
    # Load data
    colors = load_color_palette(args.cmyk_data)
    index_map = load_index_map(args.index_map)
    
    print(f"Loaded {len(colors)} colors")
    print(f"Index map dimensions: {index_map.shape}")
    
    # Create visualization
    img = create_color_visualization(
        index_map, colors, 
        cell_size=args.cell_size, 
        show_numbers=not args.no_numbers
    )
    
    # Save main visualization
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.index_map)[0]
        output_path = f"{base_name}_visualization.png"
    
    img.save(output_path)
    print(f"Visualization saved to: {output_path}")
    
    # Create legend if requested
    if args.legend:
        legend_img = create_color_palette_legend(colors)
        legend_path = f"{os.path.splitext(output_path)[0]}_legend.png"
        legend_img.save(legend_path)
        print(f"Color legend saved to: {legend_path}")

if __name__ == "__main__":
    main()
