import streamlit as st
import os
import tempfile
import zipfile
from PIL import Image
import pandas as pd
from io import BytesIO
import base64

# Import our existing functions
from cmyk_tool import (
    get_cmyk_pixels_and_counts, get_cmyk_pixels_and_counts_from_rgb_array,
    write_cmyk_data, write_index_map, write_cmyk_pixels,
    reconstruct_image_from_cmyk, quantize_image_kmeans,
    generate_visualization, rgb_to_cmyk, cmyk_to_rgb
)
import numpy as np

def process_image_streamlit(uploaded_file, scale_mode, scale_value1, scale_value2, 
                          resample_method, approx_method, precision, n_colors,
                          visualize_type, save_cmyk, reconstruct):
    """Process uploaded image with given parameters and return results."""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_image_path = tmp_file.name
    
    try:
        # Load and process image
        img = Image.open(temp_image_path).convert('RGB')
        original_size = img.size
        
        # Handle scaling
        if scale_mode == "Percentage":
            scale_factor = scale_value1 / 100
            width = int(original_size[0] * scale_factor)
            height = int(original_size[1] * scale_factor)
            scale_info = f"Scaled by {scale_value1}% from {original_size[0]}x{original_size[1]} to {width}x{height}"
        elif scale_mode == "Pixel Dimensions":
            width, height = int(scale_value1), int(scale_value2)
            scale_info = f"Scaled from {original_size[0]}x{original_size[1]} to {width}x{height}"
        else:  # Original size
            width, height = original_size
            scale_info = f"Original size: {width}x{height}"
        
        if scale_mode != "Original Size":
            resample_dict = {
                "nearest": Image.NEAREST,
                "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC,
                "lanczos": Image.LANCZOS
            }
            img = img.resize((width, height), resample_dict[resample_method])
        
        # Color reduction
        if approx_method == "precision":
            cmyk_data, color_counter, width, height = get_cmyk_pixels_and_counts(img, precision)
        elif approx_method == "kmeans":
            quantized_img = quantize_image_kmeans(img, n_colors=n_colors)
            rgb_array = np.array(quantized_img)
            cmyk_data, color_counter, width, height = get_cmyk_pixels_and_counts_from_rgb_array(rgb_array)
        elif approx_method == "median_cut":
            quantized_img = img.quantize(colors=n_colors, method=0).convert("RGB")
            rgb_array = np.array(quantized_img)
            cmyk_data, color_counter, width, height = get_cmyk_pixels_and_counts_from_rgb_array(rgb_array)
        
        # Generate results
        results = {
            'scale_info': scale_info,
            'num_colors': len(color_counter),
            'dimensions': f"{width}x{height}",
            'processed_image': img,
            'color_counter': color_counter,
            'cmyk_data': cmyk_data,
            'files': {}
        }
        
        # Create temporary directory for outputs
        temp_dir = tempfile.mkdtemp()
        base_name = os.path.join(temp_dir, "processed_image")
        
        # Generate color data CSV
        color_list = []
        with open(f"{base_name}_cmyk_data.csv", "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["Index", "Name", "C", "M", "Y", "K", "Count"])
            for idx, color in enumerate(color_counter.keys()):
                from cmyk_tool import closest_color_name, cmyk_to_rgb
                rgb = cmyk_to_rgb(*color)
                color_name = closest_color_name(rgb)
                count = color_counter[color]
                writer.writerow([idx, color_name] + [round(x, 4) for x in color] + [count])
                color_list.append(color)
        
        with open(f"{base_name}_cmyk_data.csv", "rb") as f:
            results['files']['cmyk_data'] = f.read()
        
        # Generate visualization if requested
        if visualize_type != "none":
            color_to_index = {tuple([round(x, 4) for x in color]): idx for idx, color in enumerate(color_list)}
            
            # Create index map temporarily
            index_map_file = f"{base_name}_cmyk_index_map.csv"
            with open(index_map_file, "w", newline="") as f:
                import csv
                writer = csv.writer(f)
                for row in cmyk_data:
                    index_row = []
                    for i in range(0, len(row), 4):
                        cmyk = tuple(row[i:i+4])
                        idx = color_to_index[cmyk]
                        index_row.append(idx)
                    writer.writerow(index_row)
            
            # Generate visualization
            if visualize_type == "simple":
                from cmyk_tool import load_color_palette_for_viz, load_index_map_for_viz, create_simple_visualization
                colors = load_color_palette_for_viz(f"{base_name}_cmyk_data.csv")
                index_map = load_index_map_for_viz(index_map_file)
                viz_img = create_simple_visualization(index_map, colors, cell_size=30, show_numbers=True)
                viz_img.save(f"{base_name}_visualization.png")
                
                with open(f"{base_name}_visualization.png", "rb") as f:
                    results['files']['visualization'] = f.read()
                    results['visualization_type'] = 'png'
                    
            elif visualize_type == "html":
                from cmyk_tool import create_html_visualization_integrated
                create_html_visualization_integrated(index_map_file, f"{base_name}_cmyk_data.csv", f"{base_name}_visualization.html")
                
                with open(f"{base_name}_visualization.html", "rb") as f:
                    results['files']['visualization'] = f.read()
                    results['visualization_type'] = 'html'
            
            # Clean up index map
            os.remove(index_map_file)
        
        # Generate reconstruction if requested
        if reconstruct:
            cmyk_file = f"{base_name}_cmyk.csv"
            with open(cmyk_file, "w", newline="") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow([width, height])
                writer.writerows(cmyk_data)
            
            from cmyk_tool import read_cmyk_csv
            cmyk_rows, width, height = read_cmyk_csv(cmyk_file)
            reconstructed_img = Image.new("RGB", (width, height))
            for y, row in enumerate(cmyk_rows):
                for x, cmyk in enumerate(row):
                    rgb = cmyk_to_rgb(*cmyk)
                    reconstructed_img.putpixel((x, y), rgb)
            
            reconstructed_img.save(f"{base_name}_reconstructed.png")
            
            with open(f"{base_name}_reconstructed.png", "rb") as f:
                results['files']['reconstructed'] = f.read()
            
            # Save CMYK data if requested
            if save_cmyk:
                with open(cmyk_file, "rb") as f:
                    results['files']['cmyk_pixels'] = f.read()
            
            os.remove(cmyk_file)
        
        # Clean up
        os.remove(f"{base_name}_cmyk_data.csv")
        if visualize_type == "simple" and os.path.exists(f"{base_name}_visualization.png"):
            os.remove(f"{base_name}_visualization.png")
        if visualize_type == "html" and os.path.exists(f"{base_name}_visualization.html"):
            os.remove(f"{base_name}_visualization.html")
        if reconstruct and os.path.exists(f"{base_name}_reconstructed.png"):
            os.remove(f"{base_name}_reconstructed.png")
        
        return results
        
    finally:
        # Clean up temporary image file
        os.remove(temp_image_path)

def create_download_zip(files_dict, filename_prefix="painting_parser_output"):
    """Create a ZIP file with all generated files."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_type, file_data in files_dict.items():
            if file_type == 'cmyk_data':
                zip_file.writestr(f"{filename_prefix}_cmyk_data.csv", file_data)
            elif file_type == 'visualization':
                ext = 'png' if 'visualization_type' in files_dict and files_dict['visualization_type'] == 'png' else 'html'
                zip_file.writestr(f"{filename_prefix}_visualization.{ext}", file_data)
            elif file_type == 'reconstructed':
                zip_file.writestr(f"{filename_prefix}_reconstructed.png", file_data)
            elif file_type == 'cmyk_pixels':
                zip_file.writestr(f"{filename_prefix}_cmyk.csv", file_data)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    st.set_page_config(
        page_title="PaintingParser - CMYK Color Analyzer",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 PaintingParser - CMYK Color Analyzer")
    st.markdown("Upload an image to extract and analyze its CMYK color composition!")
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Processing Parameters")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG, JPG, or JPEG image to analyze"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Size: {image.size[0]}x{image.size[1]}", width=300)
        
        with col2:
            st.subheader("ℹ️ Upload Info")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File Size:** {len(uploaded_file.getvalue())} bytes")
            st.write(f"**Image Dimensions:** {image.size[0]} x {image.size[1]} pixels")
        
        # Parameters in sidebar
        st.sidebar.subheader("🔍 Scaling Options")
        scale_mode = st.sidebar.selectbox(
            "Scaling Mode",
            ["Original Size", "Percentage", "Pixel Dimensions"],
            help="Choose how to scale the image before processing"
        )
        
        scale_value1 = scale_value2 = None
        if scale_mode == "Percentage":
            scale_value1 = st.sidebar.number_input("Scale Percentage", min_value=0.1, max_value=100.0, value=5.0, step=0.1, help="Percentage of original size")
        elif scale_mode == "Pixel Dimensions":
            col_w, col_h = st.sidebar.columns(2)
            with col_w:
                scale_value1 = st.sidebar.number_input("Width", min_value=16, max_value=2048, value=64)
            with col_h:
                scale_value2 = st.sidebar.number_input("Height", min_value=16, max_value=2048, value=64)
        
        resample_method = st.sidebar.selectbox(
            "Resampling Method",
            ["nearest", "bilinear", "bicubic", "lanczos"],
            index=0,
            help="Algorithm for image scaling quality"
        )
        
        st.sidebar.subheader("🎨 Color Reduction")
        approx_method = st.sidebar.selectbox(
            "Color Reduction Method",
            ["precision", "kmeans", "median_cut"],
            index=1,
            help="Method to reduce color complexity"
        )
        
        if approx_method == "precision":
            precision = st.sidebar.slider(
                "Precision", 0.01, 0.2, 0.05, step=0.01,
                help="Lower values = more colors, higher detail"
            )
            n_colors = None
        else:
            precision = None
            n_colors = st.sidebar.slider(
                "Number of Colors", 8, 64, 16,
                help="Number of color clusters/groups"
            )
        
        st.sidebar.subheader("📊 Output Options")
        visualize_type = st.sidebar.selectbox(
            "Visualization Type",
            ["simple", "html", "none"],
            index=0,
            help="Type of visualization to generate"
        )
        
        reconstruct = st.sidebar.checkbox(
            "Generate Reconstructed Image",
            value=False,
            help="Create a reconstructed version of the image from CMYK data"
        )
        
        save_cmyk = st.sidebar.checkbox(
            "Save Full CMYK Pixel Data",
            value=False,
            help="Save detailed CMYK data for every pixel (large file)"
        )
        
        # Process button
        if st.sidebar.button("🚀 Process Image", type="primary"):
            with st.spinner("Processing image... This may take a moment."):
                try:
                    results = process_image_streamlit(
                        uploaded_file, scale_mode, scale_value1, scale_value2,
                        resample_method, approx_method, precision, n_colors,
                        visualize_type, save_cmyk, reconstruct
                    )
                    
                    # Display results in organized sections
                    st.subheader("📊 Processing Results")
                    
                    # Results summary
                    col_result1, col_result2 = st.columns([1, 1])
                    with col_result1:
                        st.write(f"**Scaling:** {results['scale_info']}")
                        st.write(f"**Dimensions:** {results['dimensions']}")
                    with col_result2:
                        st.write(f"**Unique Colors:** {results['num_colors']}")
                        st.write(f"**Method:** {approx_method}")
                    
                    # Images section
                    col_img1, col_img2 = st.columns([1, 1])
                    
                    with col_img1:
                        # Show processed image
                        st.subheader("🖼️ Processed Image")
                        st.image(results['processed_image'], caption="Processed/Scaled Image", width=300)
                    
                    with col_img2:
                        # Display reconstructed image if available
                        if reconstruct and 'reconstructed' in results['files']:
                            st.subheader("🔄 Reconstructed Image")
                            reconstructed_image = Image.open(BytesIO(results['files']['reconstructed']))
                            st.image(reconstructed_image, caption="Reconstructed from CMYK Data", width=300)
                    
                    # Display visualization
                    if visualize_type != "none" and 'visualization' in results['files']:
                        st.subheader("🎨 Color Visualization")
                        if results.get('visualization_type') == 'png':
                            viz_image = Image.open(BytesIO(results['files']['visualization']))
                            st.image(viz_image, caption="Color Index Map", width=600)
                        elif results.get('visualization_type') == 'html':
                            st.download_button(
                                label="📄 Download HTML Visualization",
                                data=results['files']['visualization'],
                                file_name="visualization.html",
                                mime="text/html"
                            )
                    
                    # Color palette preview
                    st.subheader("🎭 Color Palette")
                    color_data = []
                    for idx, (color, count) in enumerate(results['color_counter'].items()):
                        rgb = cmyk_to_rgb(*color)
                        from cmyk_tool import closest_color_name
                        color_name = closest_color_name(rgb)
                        color_data.append({
                            'Index': idx,
                            'Color Name': color_name,
                            'C': f"{color[0]:.3f}",
                            'M': f"{color[1]:.3f}",
                            'Y': f"{color[2]:.3f}",
                            'K': f"{color[3]:.3f}",
                            'RGB': f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
                            'Count': count
                        })
                    
                    df = pd.DataFrame(color_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download section
                    st.subheader("⬇️ Download Results")
                    
                    # Individual file downloads
                    col_d1, col_d2, col_d3 = st.columns(3)
                    
                    with col_d1:
                        if 'cmyk_data' in results['files']:
                            st.download_button(
                                label="📊 Color Data CSV",
                                data=results['files']['cmyk_data'],
                                file_name="cmyk_color_data.csv",
                                mime="text/csv"
                            )
                    
                    with col_d2:
                        if 'visualization' in results['files']:
                            if results.get('visualization_type') == 'png':
                                st.download_button(
                                    label="🖼️ Visualization PNG",
                                    data=results['files']['visualization'],
                                    file_name="color_visualization.png",
                                    mime="image/png"
                                )
                    
                    with col_d3:
                        if 'reconstructed' in results['files']:
                            st.download_button(
                                label="🔄 Reconstructed Image",
                                data=results['files']['reconstructed'],
                                file_name="reconstructed_image.png",
                                mime="image/png"
                            )
                    
                    # Download all as ZIP
                    if results['files']:
                        zip_data = create_download_zip(results['files'])
                        st.download_button(
                            label="📦 Download All Files (ZIP)",
                            data=zip_data,
                            file_name="painting_parser_results.zip",
                            mime="application/zip"
                        )
                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("👆 Please upload an image file to get started!")
        
        # Instructions
        st.markdown("""
        ## 📖 How to Use
        
        1. **Upload an Image** - Choose a PNG, JPG, or JPEG file
        2. **Configure Parameters** - Adjust scaling, color reduction, and output options in the sidebar
        3. **Process** - Click the "Process Image" button
        4. **Download Results** - Get your color data, visualizations, and reconstructed images
        
        ## 🎯 Parameter Guide
        
        - **Scaling**: Control image size for processing (smaller = faster, fewer colors)
        - **Color Reduction Method**:
          - `precision`: Good general purpose method
          - `kmeans`: Best for paint-by-number style output
          - `median_cut`: Good at preserving important colors
        - **Visualization**:
          - `simple`: Creates a PNG grid with color indices
          - `html`: Creates an interactive web page
          - `none`: Skip visualization for faster processing
        """)

if __name__ == "__main__":
    main()
