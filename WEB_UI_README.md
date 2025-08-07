# PaintingParser Web UI

A user-friendly web interface for the PaintingParser tool built with Streamlit.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web application:
```bash
streamlit run streamlit_app.py
```

3. Open your browser to the URL shown in the terminal (usually http://localhost:8501)

## Features

- **Drag & Drop Image Upload**: Upload PNG, JPG, or JPEG images
- **Interactive Parameter Controls**: Adjust all processing parameters with sliders and dropdowns
- **Real-time Preview**: See original and processed images side by side
- **Color Palette Visualization**: View extracted colors in an interactive table
- **Multiple Download Options**: Download individual files or get everything in a ZIP
- **Visualization Options**: Choose between PNG grids or interactive HTML visualizations

## Usage

1. **Upload an Image**: Drag and drop or click to browse for an image file
2. **Adjust Parameters** (in sidebar):
   - **Scaling**: Choose original size, percentage scaling, or exact pixel dimensions
   - **Color Reduction**: Select method (precision, k-means, or median cut)
   - **Visualization**: Choose PNG, HTML, or skip visualization
   - **Output Options**: Enable reconstruction and/or save raw CMYK data
3. **Process**: Click the "Process Image" button
4. **View Results**: See processed image, color palette, and visualizations
5. **Download**: Get individual files or download everything as a ZIP

## Parameters Guide

### Scaling Options
- **Original Size**: Use image as-is
- **Percentage**: Scale by percentage (e.g., 50% = half size)
- **Pixel Dimensions**: Set exact width and height

### Color Reduction Methods
- **Precision**: Good general purpose (adjust precision slider)
- **K-means**: Best for paint-by-number style (adjust number of colors)
- **Median Cut**: Good at preserving important colors

### Visualization Types
- **Simple (PNG)**: Creates a color grid image
- **HTML**: Creates an interactive web page
- **None**: Skip visualization for faster processing

### Output Options
- **Generate Reconstructed Image**: Creates a version rebuilt from CMYK data (off by default)
- **Save Full CMYK Pixel Data**: Saves detailed per-pixel data (large files)

## Tips

- Start with smaller images or percentage scaling for faster processing
- Use K-means with 16-32 colors for paint-by-number projects
- Use HTML visualization for interactive color exploration
- Use precision method with low precision (0.02) for detailed color analysis
- Enable reconstruction to see how well the color reduction worked
