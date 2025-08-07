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

### For Web UI (Optional)
```bash
pip install streamlit pandas
# or install all dependencies:
pip install -r requirements.txt
```

## Web UI (Recommended for Beginners)

For a user-friendly graphical interface, use the web application:

```bash
streamlit run streamlit_app.py
```

Then open your browser to http://localhost:8501

**Features:**
- Drag & drop image upload
- Interactive parameter controls with sliders and dropdowns
- Real-time preview of results
- Download individual files or get everything in a ZIP
- No command line knowledge required!

See [WEB_UI_README.md](WEB_UI_README.md) for detailed web UI instructions.

## Command Line Usage

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
# Scale image before processing (useful for large images) - pixel dimensions
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 512 512 --approx-method kmeans --n-colors 32

# Scale by percentage (50% of original size)
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 0.5 --approx-method kmeans --n-colors 32

# Scale by percentage with different resampling method
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 0.25 --resample bilinear --approx-method kmeans --n-colors 16

# Different resampling methods with pixel dimensions
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 256 256 --resample bilinear --approx-method kmeans --n-colors 16
```

### Generate Visualizations and Reconstructions

```bash
# Generate PNG visualization (default) with paint-by-numbers style
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 16 32 --approx-method kmeans --n-colors 32

# Generate HTML visualization (interactive)
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 16 32 --approx-method kmeans --n-colors 32 --visualize html

# Skip visualization, just extract data
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 16 32 --approx-method kmeans --n-colors 32 --visualize none

# Reconstruct image from CMYK data (must specify --reconstruct)
python cmyk_tool.py --image .\SkullsAndRoses\SkullsAndRoses.png --scale 16 32 --approx-method kmeans --n-colors 32 --reconstruct

# Full workflow: extract, visualize, and reconstruct
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 512 512 --approx-method kmeans --n-colors 32 --reconstruct
```

### Save CMYK Data

```bash
# Save full CMYK pixel data with visualization
python cmyk_tool.py --image .\Wanderer\Wanderer.png --save-cmyk

# Save data without visualization
python cmyk_tool.py --image .\Wanderer\Wanderer.png --save-cmyk --visualize none

# Custom output path for reconstruction
python cmyk_tool.py --image .\Wanderer\Wanderer.png --reconstruct --output .\Wanderer\custom_output.png
```

## Visualization Options

The main tool now includes **integrated visualization** that automatically creates color visualizations and cleans up temporary files. You can choose between:

- `--visualize simple` (default) - Creates PNG visualization 
- `--visualize html` - Creates interactive HTML visualization
- `--visualize none` - Skips visualization

## Standalone Visualization Tools

*Use these tools only when you need custom visualization options or want to re-visualize existing data files.*

### HTML Visualizer

Creates an interactive HTML page with color grids and legends.

```bash
# For existing processed images (requires both CSV files)
python html_visualizer.py --index-map ".\Wanderer\Wanderer_cmyk_index_map.csv" --cmyk-data ".\Wanderer\Wanderer_cmyk_data.csv"

python html_visualizer.py --index-map ".\SkullAndRoses\SkullAndRoses_cmyk_index_map.csv" --cmyk-data ".\SkullAndRoses\SkullAndRoses_cmyk_data.csv"

# Custom output path
python html_visualizer.py --index-map ".\Wanderer\Wanderer_cmyk_index_map.csv" --cmyk-data ".\Wanderer\Wanderer_cmyk_data.csv" --output ".\Wanderer\custom_visualization.html"
```

### PNG Image Visualizer

Creates PNG images with actual colors mapped to the index grid.

```bash
# Basic PNG visualization
python simple_visualizer.py --index-map ".\Wanderer\Wanderer_cmyk_index_map.csv" --cmyk-data ".\Wanderer\Wanderer_cmyk_data.csv"

# With color legend and custom cell size
python simple_visualizer.py --index-map ".\Wanderer\Wanderer_cmyk_index_map.csv" --cmyk-data ".\Wanderer\Wanderer_cmyk_data.csv" --legend --cell-size 40

# Clean look without index numbers
python simple_visualizer.py --index-map ".\Wanderer\Wanderer_cmyk_index_map.csv" --cmyk-data ".\Wanderer\Wanderer_cmyk_data.csv" --no-numbers --legend

# Custom output path
python simple_visualizer.py --index-map ".\Wanderer\Wanderer_cmyk_index_map.csv" --cmyk-data ".\Wanderer\Wanderer_cmyk_data.csv" --output ".\Wanderer\custom_grid.png" --legend
```

## Complete Workflow Examples

### Process New Image and Visualize

```bash
# Single command: process image, reconstruct, and create PNG visualization
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 512 512 --approx-method kmeans --n-colors 32 --reconstruct

# Create HTML visualization with reconstruction
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 512 512 --approx-method kmeans --n-colors 32 --reconstruct --visualize html

# Process without reconstruction, just visualization (default)
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 512 512 --approx-method kmeans --n-colors 32
```

### Small Index Map for Paint-by-Numbers

```bash
# Create a small grid perfect for manual painting with PNG visualization
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 16 32 --approx-method kmeans --n-colors 32

# Create with reconstruction and HTML visualization for interactive color reference
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 16 32 --approx-method kmeans --n-colors 32 --reconstruct --visualize html
```

### High-Detail Analysis

```bash
# Large scale for detailed color analysis with PNG visualization
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 512 512 --approx-method precision --precision 0.02

# Create HTML visualization for detailed color exploration
python cmyk_tool.py --image .\Wanderer\Wanderer.png --scale 512 512 --approx-method precision --precision 0.02 --visualize html
```

## Scaling Options

The `--scale` parameter supports two different modes:

### Percentage Scaling
- **Single value**: Scales the image by percentage of original size
- `--scale 0.5` - Scale to 50% of original dimensions
- `--scale 0.25` - Scale to 25% of original dimensions  
- `--scale 2.0` - Scale to 200% of original dimensions
- Useful for quickly scaling images proportionally

### Pixel Dimensions
- **Two values**: Sets exact width and height in pixels
- `--scale 512 512` - Scale to exactly 512x512 pixels
- `--scale 16 32` - Scale to exactly 16x32 pixels
- `--scale 256 256` - Scale to exactly 256x256 pixels
- Useful when you need specific output dimensions

### Resampling Methods
Choose the resampling algorithm for better quality:
- `--resample nearest` (default) - Fastest, good for pixel art
- `--resample bilinear` - Good balance of speed and quality
- `--resample bicubic` - Better quality for photographic images
- `--resample lanczos` - Highest quality, slower

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

1. **For Paint-by-Numbers**: Use small scale (16x32 or 32x64) with kmeans method and PNG visualization (default)
2. **For Color Analysis**: Use larger scale (256x256+) with precision method and HTML visualization
3. **For Interactive Exploration**: Use `--visualize html` to get clickable color references
4. **For Quick Processing**: Use `--visualize none` to skip visualization when only extracting data
5. **For Printing**: PNG visualizations work well for physical reference sheets
6. **Memory Management**: Scale down large images before processing

## Example Project Structure

```
PaintingParser/
├── cmyk_tool.py              # Main command-line tool
├── html_visualizer.py        # Standalone HTML visualizer
├── simple_visualizer.py      # Standalone PNG visualizer
├── streamlit_app.py          # Web UI application
├── requirements.txt          # All dependencies
├── README.md                 # Main documentation
├── WEB_UI_README.md          # Web UI specific guide
├── Wanderer/
│   ├── Wanderer.png
│   ├── Wanderer_cmyk_data.csv
│   ├── Wanderer_reconstructed.png
│   └── Wanderer_visualization.html
└── SkullAndRoses/
    ├── SkullAndRoses.png
    ├── SkullAndRoses_cmyk_data.csv
    └── SkullAndRoses_reconstructed.png
```

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed
- **Memory issues**: Use smaller scale values for large images
- **Color naming**: Uses CSS3 color names, may approximate for unusual colors
- **Font issues**: Visualizers will fall back to default fonts if system fonts unavailable
