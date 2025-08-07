# PaintingParser

A Python tool for extracting and analyzing color information from paintings and images by converting RGB colors to CMYK color space. This tool helps understand the color composition of artwork and creates indexed color maps for easier color assembly.

## Features

- **Color Space Conversion**: Convert between RGB and CMYK color spaces
- **Color Reduction**: Multiple methods to reduce color complexity (precision, K-means, median cut)
- **Color Palette Extraction**: Extract unique colors with names and usage counts
- **Index Mapping**: Create pixel-by-pixel color index maps
- **Image Reconstruction**: Rebuild images from CMYK data
- **Visualization Tools**: Generate visual representations of color maps

## Installation

```bash
pip install pillow numpy scikit-learn webcolors==24.11.1
```

## Main Tool Usage

### Basic CMYK Color Extraction

```bash
# Extract CMYK color data from an image
python cmyk_tool.py --image .\Wanderer\Wanderer.png

# With color reduction using precision method
python cmyk_tool.py --image .\Wanderer\Wanderer.png --precision 0.05

# Using K-means clustering for color reduction
python cmyk_tool.py --image .\Wanderer\Wanderer.png --approx-method kmeans --n-colors 32
```

### Image Scaling and Processing

```bash
# Scale image before processing (useful for large images)
python cmyk_tool.py --image .\sam\sam.png --scale 512 512 --approx-method kmeans --n-colors 32

# Different resampling methods
python cmyk_tool.py --image .\sam\sam.png --scale 256 256 --resample bilinear --approx-method kmeans --n-colors 16
```

### Generate Index Maps and Reconstructions

```bash
# Create index map for color assembly
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 16 32 --approx-method kmeans --n-colors 32 --index-map

# Reconstruct image from CMYK data
python cmyk_tool.py --image .\SkullsAndRoses\SkullsAndRoses.png --scale 16 32 --approx-method kmeans --n-colors 32 --reconstruct

# Full workflow: extract, map, and reconstruct
python cmyk_tool.py --image .\sam\sam.png --scale 512 512 --approx-method kmeans --n-colors 32 --reconstruct --index-map
```

### Save CMYK Data

```bash
# Save full CMYK pixel data
python cmyk_tool.py --image .\Wanderer\Wanderer.png --save-cmyk --index-map

# Custom output path for reconstruction
python cmyk_tool.py --image .\sam\sam.png --reconstruct --output .\sam\custom_output.png
```

## Visualization Tools

### HTML Visualizer (Recommended)

Creates an interactive HTML page with color grids and legends.

```bash
# Basic HTML visualization
python html_visualizer.py --index-map ".\sam\sam_cmyk_index_map.csv" --cmyk-data ".\sam\sam_cmyk_data.csv"

# For existing processed images
python html_visualizer.py --index-map ".\Wanderer\Wanderer_cmyk_index_map.csv" --cmyk-data ".\Wanderer\Wanderer_cmyk_data.csv"

python html_visualizer.py --index-map ".\SkullAndRoses\SkullAndRoses_cmyk_index_map.csv" --cmyk-data ".\SkullAndRoses\SkullAndRoses_cmyk_data.csv"

# Custom output path
python html_visualizer.py --index-map ".\sam\sam_cmyk_index_map.csv" --cmyk-data ".\sam\sam_cmyk_data.csv" --output ".\sam\custom_visualization.html"
```

### PNG Image Visualizer

Creates PNG images with actual colors mapped to the index grid.

```bash
# Basic PNG visualization
python simple_visualizer.py --index-map ".\sam\sam_cmyk_index_map.csv" --cmyk-data ".\sam\sam_cmyk_data.csv"

# With color legend and custom cell size
python simple_visualizer.py --index-map ".\sam\sam_cmyk_index_map.csv" --cmyk-data ".\sam\sam_cmyk_data.csv" --legend --cell-size 40

# Clean look without index numbers
python simple_visualizer.py --index-map ".\sam\sam_cmyk_index_map.csv" --cmyk-data ".\sam\sam_cmyk_data.csv" --no-numbers --legend

# Custom output path
python simple_visualizer.py --index-map ".\sam\sam_cmyk_index_map.csv" --cmyk-data ".\sam\sam_cmyk_data.csv" --output ".\sam\custom_grid.png" --legend
```

## Complete Workflow Examples

### Process New Image and Visualize

```bash
# 1. Process the image and generate data files
python cmyk_tool.py --image .\sam\sam.png --scale 512 512 --approx-method kmeans --n-colors 32 --reconstruct --index-map

# 2. Create HTML visualization (interactive)
python html_visualizer.py --index-map ".\sam\sam_cmyk_index_map.csv" --cmyk-data ".\sam\sam_cmyk_data.csv"

# 3. Create PNG visualization with legend
python simple_visualizer.py --index-map ".\sam\sam_cmyk_index_map.csv" --cmyk-data ".\sam\sam_cmyk_data.csv" --legend --cell-size 30
```

### Small Index Map for Paint-by-Numbers

```bash
# Create a small grid perfect for manual painting
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 16 32 --approx-method kmeans --n-colors 32 --reconstruct --index-map

# Visualize with large cells and numbers for easy reading
python simple_visualizer.py --index-map ".\Wanderer\Wanderer_cmyk_index_map.csv" --cmyk-data ".\Wanderer\Wanderer_cmyk_data.csv" --legend --cell-size 50
```

### High-Detail Analysis

```bash
# Large scale for detailed color analysis
python cmyk_tool.py --image .\sam\sam.png --scale 512 512 --approx-method precision --precision 0.02 --index-map

# Create visualization without overwhelming detail
python html_visualizer.py --index-map ".\sam\sam_cmyk_index_map.csv" --cmyk-data ".\sam\sam_cmyk_data.csv"
```

## Color Reduction Methods

### Precision Method (Default)
- Quantizes CMYK values to specified precision
- `--precision 0.05` (default) - good balance
- `--precision 0.02` - more colors, higher detail
- `--precision 0.1` - fewer colors, more simplified

### K-means Clustering
- Uses machine learning to group similar colors
- `--approx-method kmeans --n-colors 32` - 32 color clusters
- `--approx-method kmeans --n-colors 16` - simpler palette
- Best for creating paint-by-number style images

### Median Cut
- Uses PIL's built-in quantization
- `--approx-method median_cut --n-colors 24`
- Good for preserving important colors

## Output Files

When you run the main tool, it generates:

- `*_cmyk_data.csv` - Color palette with indices, names, CMYK values, and pixel counts
- `*_cmyk_index_map.csv` - Grid showing which color index is used at each pixel
- `*_reconstructed.png` - Reconstructed image from CMYK data (if --reconstruct used)
- `*_cmyk.csv` - Full pixel CMYK data (if --save-cmyk used)

When you run visualizers:

- `*_visualization.html` - Interactive HTML color map
- `*_visualization.png` - PNG image of color grid
- `*_legend.png` - Color palette legend (if --legend used)

## Tips for Best Results

1. **For Paint-by-Numbers**: Use small scale (16x32 or 32x64) with kmeans method
2. **For Color Analysis**: Use larger scale (256x256+) with precision method
3. **For Web Viewing**: HTML visualizer works in any browser
4. **For Printing**: PNG visualizer with large cell sizes (40-50px)
5. **Memory Management**: Scale down large images before processing

## Example Project Structure

```
PaintingParser/
├── cmyk_tool.py
├── html_visualizer.py
├── simple_visualizer.py
├── Wanderer/
│   ├── Wanderer.png
│   ├── Wanderer_cmyk_data.csv
│   ├── Wanderer_cmyk_index_map.csv
│   ├── Wanderer_reconstructed.png
│   └── Wanderer_visualization.html
└── sam/
    ├── sam.png
    ├── sam_cmyk_data.csv
    ├── sam_cmyk_index_map.csv
    └── sam_reconstructed.png
```

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed
- **Memory issues**: Use smaller scale values for large images
- **Color naming**: Uses CSS3 color names, may approximate for unusual colors
- **Font issues**: Visualizers will fall back to default fonts if system fonts unavailable
