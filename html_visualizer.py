import csv
import argparse

def cmyk_to_rgb(c, m, y, k):
    """Convert CMYK to RGB."""
    r = int(255 * (1 - c) * (1 - k))
    g = int(255 * (1 - m) * (1 - k))
    b = int(255 * (1 - y) * (1 - k))
    return r, g, b

def create_html_visualization(index_map_file, cmyk_data_file, output_file):
    """Create an HTML file with color visualization."""
    
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
            line-height: 25px;
            color: white;
            text-shadow: 1px 1px 1px black;
        }
        .legend { margin-top: 20px; }
        .legend-item { 
            display: inline-block; 
            margin: 5px; 
            padding: 5px; 
            border: 1px solid black;
            vertical-align: top;
        }
        .color-square { 
            width: 30px; height: 30px; 
            display: inline-block; 
            margin-right: 10px;
            border: 1px solid black;
        }
    </style>
</head>
<body>
    <h1>CMYK Index Map Visualization</h1>
    
    <h2>Color Grid</h2>
    <div class="grid">
"""
    
    for row in index_map:
        html += "        <div>\n"
        for index in row:
            if index in colors:
                color = colors[index]
                html += f'            <div class="cell" style="background-color: {color["rgb"]};" title="Index {index}: {color["name"]} - {color["cmyk"]}">{index}</div>\n'
            else:
                html += f'            <div class="cell" style="background-color: magenta;" title="Missing color index {index}">{index}</div>\n'
        html += "        </div>\n"
    
    html += """    </div>
    
    <h2>Color Legend</h2>
    <div class="legend">
"""
    
    for index in sorted(colors.keys()):
        color = colors[index]
        html += f"""        <div class="legend-item">
            <div class="color-square" style="background-color: {color['rgb']};"></div>
            <strong>Index {index}:</strong> {color['name']}<br>
            RGB: {color['rgb']}<br>
            Hex: {color['hex']}<br>
            CMYK: {color['cmyk']}<br>
            Count: {color['count']} pixels
        </div>
"""
    
    html += """    </div>
</body>
</html>"""
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"HTML visualization saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create HTML visualization of CMYK index map")
    parser.add_argument("--index-map", required=True, help="Path to the index map CSV file")
    parser.add_argument("--cmyk-data", required=True, help="Path to the CMYK data CSV file")
    parser.add_argument("--output", help="Output HTML file path")
    
    args = parser.parse_args()
    
    if args.output:
        output_file = args.output
    else:
        base_name = args.index_map.replace('_cmyk_index_map.csv', '')
        output_file = f"{base_name}_visualization.html"
    
    create_html_visualization(args.index_map, args.cmyk_data, output_file)

if __name__ == "__main__":
    main()
