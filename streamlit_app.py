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
import math

def process_image_streamlit(uploaded_file, scale_mode, scale_value1, scale_value2,
                             resample_method, approx_method, precision, n_colors,
                             visualize_type, save_cmyk, reconstruct):
    """Process uploaded image with given parameters and return results."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_image_path = tmp_file.name

    try:
        img = Image.open(temp_image_path).convert('RGB')
        orig_w, orig_h = img.size
        if scale_mode == "Percentage":
            sf = scale_value1 / 100
            width = max(1, int(orig_w * sf))
            height = max(1, int(orig_h * sf))
            scale_info = f"Scaled by {scale_value1}% from {orig_w}x{orig_h} to {width}x{height}"
        elif scale_mode == "Original Size":
            width, height = orig_w, orig_h
            scale_info = f"Original size: {width}x{height}"
        else:
            width = max(1, int(scale_value1))
            height = max(1, int(scale_value2))
            scale_info = f"Scaled from {orig_w}x{orig_h} to {width}x{height}"
        if scale_mode != "Original Size":
            resample = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}[resample_method]
            img = img.resize((width, height), resample)

        if approx_method == "precision":
            cmyk_data, color_counter, width, height = get_cmyk_pixels_and_counts(img, precision)
        elif approx_method == "kmeans":
            qimg = quantize_image_kmeans(img, n_colors=n_colors)
            rgb_array = np.array(qimg)
            cmyk_data, color_counter, width, height = get_cmyk_pixels_and_counts_from_rgb_array(rgb_array)
        elif approx_method == "median_cut":
            qimg = img.quantize(colors=n_colors, method=0).convert("RGB")
            rgb_array = np.array(qimg)
            cmyk_data, color_counter, width, height = get_cmyk_pixels_and_counts_from_rgb_array(rgb_array)
        else:
            raise ValueError(f"Unknown color reduction method: {approx_method}")

        results = {
            'scale_info': scale_info,
            'num_colors': len(color_counter),
            'dimensions': f"{width}x{height}",
            'processed_image': img,
            'color_counter': color_counter,
            'cmyk_data': cmyk_data,
            'approx_method': approx_method,
            'files': {},
            'color_list': []
        }

        temp_dir = tempfile.mkdtemp()
        base_name = os.path.join(temp_dir, "processed_image")

        # Deterministic ordering: count desc then CMYK tuple
        sorted_colors = sorted(color_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        color_list = []
        import csv
        with open(f"{base_name}_cmyk_data.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Index", "Name", "C", "M", "Y", "K", "Count"])
            for idx, (color, cnt) in enumerate(sorted_colors):
                from cmyk_tool import closest_color_name, cmyk_to_rgb
                rgb = cmyk_to_rgb(*color)
                w.writerow([idx, closest_color_name(rgb)] + [round(x,4) for x in color] + [cnt])
                color_list.append(color)
        with open(f"{base_name}_cmyk_data.csv", "rb") as f:
            results['files']['cmyk_data'] = f.read()
        results['color_list'] = color_list

        if visualize_type != "none":
            color_to_index = {tuple([round(x,4) for x in c]): i for i, c in enumerate(color_list)}
            index_map_file = f"{base_name}_cmyk_index_map.csv"
            with open(index_map_file, "w", newline="") as f:
                w = csv.writer(f)
                for row in cmyk_data:
                    idx_row = []
                    for i in range(0, len(row), 4):
                        cmyk = tuple(row[i:i+4])
                        idx_row.append(color_to_index[cmyk])
                    w.writerow(idx_row)
            if visualize_type == "simple":
                from cmyk_tool import load_color_palette_for_viz, load_index_map_for_viz, create_simple_visualization
                colors = load_color_palette_for_viz(f"{base_name}_cmyk_data.csv")
                idx_map = load_index_map_for_viz(index_map_file)
                viz_img = create_simple_visualization(idx_map, colors, cell_size=30, show_numbers=True)
                viz_path = f"{base_name}_visualization.png"
                viz_img.save(viz_path)
                with open(viz_path, 'rb') as vf:
                    results['files']['visualization'] = vf.read()
                    results['visualization_type'] = 'png'
            else:
                from cmyk_tool import create_html_visualization_integrated
                html_path = f"{base_name}_visualization.html"
                create_html_visualization_integrated(index_map_file, f"{base_name}_cmyk_data.csv", html_path)
                with open(html_path, 'rb') as hf:
                    results['files']['visualization'] = hf.read()
                    results['visualization_type'] = 'html'
            try:
                os.remove(index_map_file)
            except Exception:
                pass

        if reconstruct:
            import csv as _csv
            cmyk_file = f"{base_name}_cmyk.csv"
            with open(cmyk_file, 'w', newline='') as f:
                w = _csv.writer(f)
                w.writerow([width, height])
                w.writerows(cmyk_data)
            from cmyk_tool import read_cmyk_csv
            cmyk_rows, width, height = read_cmyk_csv(cmyk_file)
            recon = Image.new('RGB', (width, height))
            for y, row in enumerate(cmyk_rows):
                for x, cmyk in enumerate(row):
                    recon.putpixel((x,y), cmyk_to_rgb(*cmyk))
            recon_path = f"{base_name}_reconstructed.png"
            recon.save(recon_path)
            with open(recon_path, 'rb') as rf:
                results['files']['reconstructed'] = rf.read()
            if save_cmyk:
                with open(cmyk_file, 'rb') as cf:
                    results['files']['cmyk_pixels'] = cf.read()
            for p in (cmyk_file, recon_path):
                try: os.remove(p)
                except Exception: pass

        # cleanup temp artifacts
        for suffix in ('_cmyk_data.csv', '_visualization.png', '_visualization.html'):
            p = f"{base_name}{suffix}"
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        return results
    finally:
        try:
            os.remove(temp_image_path)
        except Exception:
            pass

def create_download_zip(files_dict, visualization_type=None, filename_prefix="painting_parser_output"):
    """Create a ZIP file with all generated files.
    visualization_type: optional hint ('png'|'html') because files_dict keeps raw bytes only.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_type, file_data in files_dict.items():
            if file_type == 'cmyk_data':
                zip_file.writestr(f"{filename_prefix}_cmyk_data.csv", file_data)
            elif file_type == 'mixing_data':
                zip_file.writestr(f"{filename_prefix}_cmyk_mixing_recipes.csv", file_data)
            elif file_type == 'visualization':
                # Determine extension robustly
                ext = visualization_type
                if ext is None:
                    if file_data.startswith(b'\x89PNG'):
                        ext = 'png'
                    elif file_data.lstrip().lower().startswith(b'<!doctype') or file_data.lstrip().startswith(b'<html'):
                        ext = 'html'
                    else:
                        ext = 'dat'
                zip_file.writestr(f"{filename_prefix}_visualization.{ext}", file_data)
            elif file_type == 'reconstructed':
                zip_file.writestr(f"{filename_prefix}_reconstructed.png", file_data)
            elif file_type == 'cmyk_pixels':
                zip_file.writestr(f"{filename_prefix}_cmyk.csv", file_data)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_print_swatch_sheet(color_data, results, page_size="a4", swatches_per_row=8, output_format="png", square_size_inches=0.2, include_visualization=False):
    """Create a printable swatch sheet with color swatches for physical comparison.
    
    Args:
        color_data: List of color dictionaries with RGB, CMYK, names, etc.
        results: Processing results dict
        page_size: "letter" (8.5x11) or "a4" 
        swatches_per_row: Number of swatches per row
        output_format: "png" or "pdf"
        square_size_inches: Size of color squares in inches
        include_visualization: Whether to include the color visualization on the sheet
    
    Returns:
        PIL Image (for PNG) or bytes (for PDF)
    """
    
    if output_format == "pdf":
        return create_pdf_swatch_sheet(color_data, results, page_size, swatches_per_row, square_size_inches, include_visualization)
    
    # PNG generation (existing code)
    # Page dimensions in pixels at 300 DPI for crisp printing
    if page_size == "letter":
        page_width, page_height = 2550, 3300  # 8.5" x 11" at 300 DPI
    else:  # A4
        page_width, page_height = 2480, 3508  # 8.27" x 11.69" at 300 DPI
    
    # Calculate layout
    margin = 150  # 0.5" margin at 300 DPI
    usable_width = page_width - (2 * margin)
    usable_height = page_height - (2 * margin)
    
    # Header space for title and image info
    header_height = 400  # Increased for visualization
    content_start_y = margin + header_height
    content_height = usable_height - header_height
    
    # Check if we have visualization to include
    has_visualization = include_visualization and 'visualization' in results['files'] and results.get('visualization_type') == 'png'
    
    # Swatch layout - configurable square size
    swatch_gap = 15
    
    # Reserve space for visualization if available (left side)
    viz_width = 0
    if has_visualization:
        viz_width = min(600, usable_width // 3)  # Use up to 1/3 of width
        available_swatch_width = usable_width - viz_width - 30  # 30px gap
    else:
        available_swatch_width = usable_width
    
    column_width = (available_swatch_width - (swatches_per_row - 1) * swatch_gap) // swatches_per_row
    
    # Convert square size from inches to pixels at 300 DPI
    color_square_size = int(square_size_inches * 300)  # 300 DPI conversion
    
    # Ensure square fits in column, but don't make it smaller than requested unless necessary
    if color_square_size > column_width * 0.8:  # Leave some space for text alignment
        color_square_size = int(column_width * 0.8)
    
    text_area_height = 200  # Plenty of space for larger text
    swatch_height = color_square_size + text_area_height
    
    rows_needed = math.ceil(len(color_data) / swatches_per_row)
    
    # Create white background
    img = Image.new('RGB', (page_width, page_height), 'white')
    
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Larger, more readable fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 48)
            info_font = ImageFont.truetype("arial.ttf", 28)
            swatch_font = ImageFont.truetype("arial.ttf", 20)  # Larger for readability
            small_font = ImageFont.truetype("arial.ttf", 16)   # Larger than before
            tiny_font = ImageFont.truetype("arial.ttf", 14)    # Still readable
        except:
            try:
                title_font = ImageFont.truetype("calibri.ttf", 48)
                info_font = ImageFont.truetype("calibri.ttf", 28)
                swatch_font = ImageFont.truetype("calibri.ttf", 20)
                small_font = ImageFont.truetype("calibri.ttf", 16)
                tiny_font = ImageFont.truetype("calibri.ttf", 14)
            except:
                # Fall back to default font
                title_font = ImageFont.load_default()
                info_font = ImageFont.load_default()
                swatch_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
                tiny_font = ImageFont.load_default()
        
        # Header
        title = "🎨 PaintingParser - Color Swatch Reference Sheet"
        draw.text((margin, margin), title, fill='black', font=title_font)
        
        # Image info
        info_text = f"Image: {results.get('scale_info', 'N/A')} | Colors: {len(color_data)} | Method: {results.get('approx_method', 'N/A')}"
        draw.text((margin, margin + 60), info_text, fill='black', font=info_font)
        
        # Print date and settings
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        settings_text = f"Generated: {timestamp} | Print at 300 DPI for actual size"
        draw.text((margin, margin + 100), settings_text, fill='gray', font=small_font)
        
        # Total pixels for percentage calculations
        total_pixels = results['processed_image'].size[0] * results['processed_image'].size[1]
        pixels_text = f"Total Image Pixels: {total_pixels:,}"
        draw.text((margin, margin + 130), pixels_text, fill='gray', font=small_font)
        
        # Add visualization if available
        viz_end_x = margin
        if has_visualization:
            try:
                viz_img = Image.open(BytesIO(results['files']['visualization']))
                # Scale visualization to fit reserved space
                viz_ratio = min(viz_width / viz_img.width, (content_height * 0.8) / viz_img.height)
                viz_new_width = int(viz_img.width * viz_ratio)
                viz_new_height = int(viz_img.height * viz_ratio)
                viz_resized = viz_img.resize((viz_new_width, viz_new_height), Image.LANCZOS)
                
                # Position visualization
                viz_x = margin
                viz_y = content_start_y
                img.paste(viz_resized, (viz_x, viz_y))
                
                # Add visualization label
                viz_label = "Color Index Visualization"
                draw.text((viz_x, viz_y - 25), viz_label, fill='black', font=small_font)
                
                viz_end_x = viz_x + viz_new_width + 30  # 30px gap
                
            except Exception as e:
                # If visualization fails, just skip it
                pass
        
        # Calculate swatch start position (after visualization if present)
        swatch_start_x = viz_end_x
        
        # Color swatches
        for i, color_entry in enumerate(color_data):
            row = i // swatches_per_row
            col = i % swatches_per_row
            
            # Calculate column position (starting after visualization)
            column_x = swatch_start_x + col * (column_width + swatch_gap)
            y = content_start_y + row * (swatch_height + swatch_gap)
            
            # Skip if we're running out of vertical space
            if y + swatch_height > page_height - margin:
                break
            
            # Extract RGB color
            rgb_str = color_entry['RGB']  # format: "rgb(r,g,b)"
            r, g, b = [int(x) for x in rgb_str[4:-1].split(',')]
            
            # Draw color square (centered in column)
            square_x = column_x + (column_width - color_square_size) // 2
            square_rect = [square_x, y, square_x + color_square_size, y + color_square_size]
            draw.rectangle(square_rect, fill=(r, g, b), outline='black', width=2)
            
            # Text below square - also centered in column for visual alignment
            text_start_x = column_x + (column_width - color_square_size) // 2  # Align with square
            text_y = y + color_square_size + 8
            line_height = 22  # More generous line spacing
            
            # Index and name (line 1) - larger, bold-looking
            index_name = f"#{color_entry['Index']} {color_entry['Color Name'][:12]}"
            draw.text((text_start_x, text_y), index_name, fill='black', font=swatch_font)
            text_y += line_height + 3
            
            # CMYK percentages (line 2) - most important info, larger
            cmyk_text = f"C{color_entry['C%']} M{color_entry['M%']} Y{color_entry['Y%']} K{color_entry['K%']}"
            draw.text((text_start_x, text_y), cmyk_text, fill='black', font=small_font)
            text_y += line_height
            
            # CMYK decimal values (line 3)
            cmyk_decimal = f"({color_entry['C']}, {color_entry['M']}, {color_entry['Y']}, {color_entry['K']})"
            draw.text((text_start_x, text_y), cmyk_decimal, fill='blue', font=tiny_font)
            text_y += line_height - 2
            
            # RGB values (line 4)
            rgb_text = f"RGB({r},{g},{b})"
            draw.text((text_start_x, text_y), rgb_text, fill='gray', font=tiny_font)
            text_y += line_height - 2
            
            # Pixel count and percentage (line 5)
            pixel_count = color_entry['Count']
            pixel_pct = (pixel_count / total_pixels) * 100
            pixel_text = f"{pixel_count:,}px ({pixel_pct:.1f}%)"
            draw.text((text_start_x, text_y), pixel_text, fill='darkgreen', font=small_font)
            text_y += line_height - 2
            
            # Mixing parts if available (line 6)
            if color_entry.get('CMYK Parts'):
                parts_text = color_entry['CMYK Parts'][:15]  # Allow more characters
                draw.text((text_start_x, text_y), parts_text, fill='purple', font=tiny_font)
        
        # Footer with printing instructions and legend
        footer_y = page_height - margin - 80
        footer_text = "Print Settings: 300 DPI, Actual Size, Color Mode. Compare small squares under your painting light."
        draw.text((margin, footer_y), footer_text, fill='gray', font=small_font)
        
        # Legend
        legend_y = footer_y + 25
        legend_text = "Legend: Black=Index/Name & CMYK%, Blue=CMYK decimals, Gray=RGB, Green=Pixels, Purple=Mix ratios"
        draw.text((margin, legend_y), legend_text, fill='gray', font=tiny_font)
        
    except ImportError:
        # Fallback if PIL doesn't have ImageDraw/ImageFont
        # Create a simpler version with just small color blocks
        pixels = img.load()
        
        # Fill with small color swatches in a grid
        for i, color_entry in enumerate(color_data):
            if i >= swatches_per_row * 10:  # Limit to avoid too many swatches
                break
                
            row = i // swatches_per_row
            col = i % swatches_per_row
            
            x_start = margin + col * (column_width + swatch_gap) + (column_width - color_square_size) // 2
            y_start = content_start_y + row * (swatch_height + swatch_gap)
            
            rgb_str = color_entry['RGB']
            r, g, b = [int(x) for x in rgb_str[4:-1].split(',')]
            
            # Fill small color rectangle
            for y in range(y_start, min(y_start + color_square_size, page_height)):
                for x in range(x_start, min(x_start + color_square_size, page_width)):
                    if 0 <= x < page_width and 0 <= y < page_height:
                        pixels[x, y] = (r, g, b)
    
    return img

def create_pdf_swatch_sheet(color_data, results, page_size="a4", swatches_per_row=8, square_size_inches=0.2, include_visualization=False):
    """Create a PDF swatch sheet using ReportLab."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.units import inch
        from reportlab.lib.colors import Color
        import io
        
        # Set up page dimensions
        if page_size == "letter":
            page_width, page_height = letter
        else:  # A4
            page_width, page_height = A4
        
        # Create PDF in memory
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=(page_width, page_height))
        
        # Margins and layout
        margin = 0.5 * inch
        usable_width = page_width - (2 * margin)
        usable_height = page_height - (2 * margin)
        
        # Header space
        header_height = 1.4 * inch  # Increased for visualization
        content_start_y = page_height - margin - header_height
        
        # Check if we have visualization to include
        has_visualization = include_visualization and 'visualization' in results['files'] and results.get('visualization_type') == 'png'
        
        # Reserve space for visualization if available
        viz_width = 0
        if has_visualization:
            viz_width = min(2.5 * inch, usable_width / 3)  # Use up to 1/3 of width
            available_swatch_width = usable_width - viz_width - 0.2 * inch  # 0.2" gap
        else:
            available_swatch_width = usable_width
        
        # Calculate swatch layout with configurable square size
        swatch_gap = 0.05 * inch
        column_width = (available_swatch_width - (swatches_per_row - 1) * swatch_gap) / swatches_per_row
        
        # Use the specified square size in inches
        color_square_size = square_size_inches * inch
        
        # Ensure square fits in column
        if color_square_size > column_width * 0.8:
            color_square_size = column_width * 0.8
        
        # Increased text area height for better spacing
        text_area_height = 1.2 * inch  # Increased from 0.8"
        swatch_height = color_square_size + text_area_height
        
        # Header
        c.setFillColor(Color(0, 0, 0))  # Black text
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, page_height - margin - 0.3 * inch, "🎨 PaintingParser - Color Swatch Reference Sheet")
        
        # Image info
        c.setFont("Helvetica", 10)
        info_text = f"Image: {results.get('scale_info', 'N/A')} | Colors: {len(color_data)} | Method: {results.get('approx_method', 'N/A')}"
        c.drawString(margin, page_height - margin - 0.5 * inch, info_text)
        
        # Timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        settings_text = f"Generated: {timestamp} | Print at actual size"
        c.setFont("Helvetica", 8)
        c.drawString(margin, page_height - margin - 0.7 * inch, settings_text)
        
        # Total pixels
        total_pixels = results['processed_image'].size[0] * results['processed_image'].size[1]
        pixels_text = f"Total Image Pixels: {total_pixels:,}"
        c.drawString(margin, page_height - margin - 0.9 * inch, pixels_text)
        
        # Add visualization if available
        viz_end_x = margin
        if has_visualization:
            try:
                from reportlab.lib.utils import ImageReader
                viz_img = Image.open(BytesIO(results['files']['visualization']))
                
                # Scale visualization to fit reserved space
                viz_ratio = min(viz_width / viz_img.width, (content_start_y - margin - 1.0 * inch) / viz_img.height)
                viz_new_width = viz_img.width * viz_ratio
                viz_new_height = viz_img.height * viz_ratio
                
                # Position visualization
                viz_x = margin
                viz_y = content_start_y - viz_new_height
                
                # Draw visualization
                c.drawImage(ImageReader(viz_img), viz_x, viz_y, viz_new_width, viz_new_height)
                
                # Add visualization label
                c.setFillColor(Color(0, 0, 0))  # Black text
                c.setFont("Helvetica-Bold", 8)
                c.drawString(viz_x, viz_y + viz_new_height + 0.1 * inch, "Color Index Visualization")
                
                viz_end_x = viz_x + viz_new_width + 0.2 * inch  # 0.2" gap
                
            except Exception as e:
                # If visualization fails, just skip it
                pass
        
        # Color swatches
        for i, color_entry in enumerate(color_data):
            row = i // swatches_per_row
            col = i % swatches_per_row
            
            # Calculate position (PDF coordinates are bottom-up)
            x = viz_end_x + col * (column_width + swatch_gap)
            y = content_start_y - row * (swatch_height + swatch_gap)
            
            # Skip if we're running out of vertical space
            if y - swatch_height < margin:
                break
            
            # Extract RGB color
            rgb_str = color_entry['RGB']  # format: "rgb(r,g,b)"
            r, g, b = [int(x) for x in rgb_str[4:-1].split(',')]
            
            # Draw color square (centered in column)
            square_x = x + (column_width - color_square_size) / 2
            square_y = y - color_square_size
            
            # Set fill color and draw rectangle
            c.setFillColor(Color(r/255.0, g/255.0, b/255.0))
            c.setStrokeColor(Color(0, 0, 0))  # Black border
            c.setLineWidth(1)
            c.rect(square_x, square_y, color_square_size, color_square_size, fill=1, stroke=1)
            
            # Text below square - all black text with proper spacing
            text_x = x + (column_width - color_square_size) / 2
            text_y = square_y - 0.15 * inch  # More space from square
            line_height = 0.15 * inch  # Increased line spacing
            
            # All text is black
            c.setFillColor(Color(0, 0, 0))
            
            # Index and name (line 1)
            c.setFont("Helvetica-Bold", 9)  # Slightly larger
            index_name = f"#{color_entry['Index']} {color_entry['Color Name'][:10]}"
            c.drawString(text_x, text_y, index_name)
            text_y -= line_height
            
            # CMYK percentages (line 2)
            c.setFont("Helvetica-Bold", 8)
            cmyk_text = f"C{color_entry['C%']} M{color_entry['M%']} Y{color_entry['Y%']} K{color_entry['K%']}"
            c.drawString(text_x, text_y, cmyk_text)
            text_y -= line_height
            
            # CMYK decimal values (line 3)
            c.setFont("Helvetica", 7)
            cmyk_decimal = f"({color_entry['C']}, {color_entry['M']}, {color_entry['Y']}, {color_entry['K']})"
            c.drawString(text_x, text_y, cmyk_decimal)
            text_y -= line_height
            
            # RGB values (line 4)
            c.setFont("Helvetica", 7)
            rgb_text = f"RGB({r},{g},{b})"
            c.drawString(text_x, text_y, rgb_text)
            text_y -= line_height
            
            # Pixel count and percentage (line 5)
            c.setFont("Helvetica-Bold", 7)
            pixel_count = color_entry['Count']
            pixel_pct = (pixel_count / total_pixels) * 100
            pixel_text = f"{pixel_count:,}px ({pixel_pct:.1f}%)"
            c.drawString(text_x, text_y, pixel_text)
            text_y -= line_height
            
            # Mixing parts if available (line 6)
            if color_entry.get('CMYK Parts'):
                c.setFont("Helvetica", 7)
                parts_text = color_entry['CMYK Parts'][:12]  # Truncate to fit
                c.drawString(text_x, text_y, parts_text)
        
        # Footer - all black text
        c.setFillColor(Color(0, 0, 0))
        c.setFont("Helvetica", 8)
        footer_text = "Print Settings: Actual Size, Color Mode. Compare squares under your painting light."
        c.drawString(margin, margin + 0.3 * inch, footer_text)
        
        # Legend
        c.setFont("Helvetica", 7)
        legend_text = "All values shown: Index/Name, CMYK percentages, CMYK decimals, RGB values, Pixel count, Mix ratios"
        c.drawString(margin, margin + 0.1 * inch, legend_text)
        
        # Finalize PDF
        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
        
    except ImportError:
        # Fallback if ReportLab is not available
        st.error("PDF generation requires the 'reportlab' library. Install with: pip install reportlab")
        return None

def main():
    st.set_page_config(page_title="PaintingParser - CMYK Color Analyzer", page_icon="🎨", layout="wide")

    st.title("🎨 PaintingParser - CMYK Color Analyzer")
    st.markdown("Upload an image to extract and analyze its CMYK color composition!")
    # Global CSS adjustments for header spacing
    st.markdown(
        """
<style>
/* Increase vertical spacing around headers for clearer section separation */
main h1 {margin-top:0.2rem !important; margin-bottom:1.2rem !important;}
main h2, main h3, main h4 {margin-top:2.0rem !important; margin-bottom:0.75rem !important;}
/* Tighter spacing for consecutive headers */
main h2 + h3, main h3 + h4 {margin-top:0.9rem !important;}
/* Adjust expander header spacing */
div.streamlit-expanderHeader p {margin:0.4rem 0 !important;}
/* Slightly more breathing room at page sides */
section.main > div.block-container {padding-top:1.2rem !important;}
</style>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.header("⚙️ Processing Parameters")
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'], help="Upload a PNG, JPG, or JPEG image to analyze")

    # Default to bundled SkullAndRoses sample if user hasn't uploaded anything yet
    sample_default = False
    if uploaded_file is None:
        sample_path = os.path.join(os.path.dirname(__file__), 'SkullAndRoses', 'SkullAndRoses.png')
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                sample_bytes = f.read()
            uploaded_file = BytesIO(sample_bytes)
            uploaded_file.name = 'SkullAndRoses.png'
            sample_default = True
            st.info("Using bundled sample image: SkullAndRoses.png (upload your own to replace)")

    def build_and_render(results, approx_method, reconstruct, visualize_type):
        # Summary
        st.subheader("📊 Processing Results")
        c1, c2 = st.columns([1,1])
        with c1:
            st.write(f"**Scaling:** {results['scale_info']}")
            st.write(f"**Dimensions:** {results['dimensions']}")
        with c2:
            st.write(f"**Unique Colors:** {results['num_colors']}")
            st.write(f"**Method:** {approx_method}")

        # Images
        img_c1, img_c2 = st.columns([1,1])
        with img_c1:
            st.subheader("🖼️ Processed Image")
            st.image(results['processed_image'], caption="Processed/Scaled Image", width=300)
        with img_c2:
            if reconstruct and 'reconstructed' in results['files']:
                st.subheader("🔄 Reconstructed Image")
                r_img = Image.open(BytesIO(results['files']['reconstructed']))
                st.image(r_img, caption="Reconstructed from CMYK Data", width=300)

        # Visualization
        if visualize_type != 'none' and 'visualization' in results['files']:
            st.subheader("🎨 Color Visualization")
            if results.get('visualization_type') == 'png':
                v_img = Image.open(BytesIO(results['files']['visualization']))
                st.image(v_img, caption="Color Index Map", width=600)
            else:
                st.download_button("📄 Download HTML Visualization", results['files']['visualization'], 'visualization.html', 'text/html')

        # Color data + mixing prep (shared by expander and explorer)
        def simplify_parts(parts_dict):
            vals = [v for v in parts_dict.values() if v>0]
            if not vals: return parts_dict
            g = 0
            for v in vals: g = v if g==0 else math.gcd(g, v)
            if g <= 1: return parts_dict
            return {k:(v//g if v>0 else 0) for k,v in parts_dict.items()}
        def mixing_recipe(c,m,y,k):
            Cp, Mp, Yp, Kp = [int(round(x*100)) for x in (c,m,y,k)]
            full = simplify_parts({'C':Cp,'M':Mp,'Y':Yp,'K':Kp})
            cmy = simplify_parts({'C':Cp,'M':Mp,'Y':Yp})
            join = lambda d: ' + '.join(f"{v}{k}" for k,v in d.items() if v>0) if any(v>0 for v in d.values()) else '—'
            return {'C%':Cp,'M%':Mp,'Y%':Yp,'K%':Kp,'CMY Parts':join(cmy),'CMYK Parts':join(full)}

        color_data = []
        mixing_rows = []
        # Build deterministically using the saved color_list (sorted by count desc),
        # so Index values are stable and match the CSV/visualization.
        ordered_colors = results.get('color_list') or list(results['color_counter'].keys())
        for idx, color in enumerate(ordered_colors):
            count = results['color_counter'].get(color, 0)
            rgb = cmyk_to_rgb(*color)
            from cmyk_tool import closest_color_name
            name = closest_color_name(rgb)
            rec = mixing_recipe(*color)
            cd = {
                'Index': idx,
                'Color Name': name,
                'C': f"{color[0]:.3f}", 'M': f"{color[1]:.3f}", 'Y': f"{color[2]:.3f}", 'K': f"{color[3]:.3f}",
                'C%': rec['C%'], 'M%': rec['M%'], 'Y%': rec['Y%'], 'K%': rec['K%'],
                'CMY Parts': rec['CMY Parts'], 'CMYK Parts': rec['CMYK Parts'],
                'RGB': f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", 'Count': count
            }
            color_data.append(cd)
            mixing_rows.append({
                'Index': idx, 'Color Name': name,
                'C': color[0], 'M': color[1], 'Y': color[2], 'K': color[3],
                'C%': rec['C%'], 'M%': rec['M%'], 'Y%': rec['Y%'], 'K%': rec['K%'],
                'CMY Parts': rec['CMY Parts'], 'CMYK Parts': rec['CMYK Parts'],
                'Count': count
            })

        if 'mixing_data' not in results['files']:
            results['files']['mixing_data'] = pd.DataFrame(mixing_rows).to_csv(index=False).encode('utf-8')

        # Keep an immutable base index map for stable selection lookup across different view orderings
        base_by_index = {cd['Index']: cd for cd in color_data}

        # Collapsible palette table only
        with st.expander("🎭 Color Palette (Table)", expanded=False):
            st.dataframe(pd.DataFrame(color_data), use_container_width=True)
            st.caption("Click swatches in the explorer below to inspect a color.")
 
        # Large color count advisory
        if results['num_colors'] > 512:
            st.info(f"High number of unique colors ({results['num_colors']}). Consider stronger reduction (higher precision / fewer kmeans colors) for easier exploration.")

        # Color ordering control (add mix-friendly path ordering)
        order_key = 'pp_color_order'
        # First time: default to Mix Path (additive progression) instead of count
        default_order = st.session_state.get(order_key, 'Mix Path')
        ordering_options = ['Mix Path','Count (desc)','Hue','Luminance','Cyan %','Magenta %','Yellow %','Black %','Index']
        # Safety: if session has legacy value not in list
        if default_order not in ordering_options:
            default_order = 'Mix Path'
        order_choice = st.selectbox(
            'Palette Ordering',
            ordering_options,
            index=ordering_options.index(default_order),
            help='Ordering of swatches. Mix Path attempts an additive sequence where each next color adds ink without removing.'
        )
        st.session_state[order_key] = order_choice

        # Selection state
        if 'selected_color' not in st.session_state:
            st.session_state['selected_color'] = None
        selected_idx = st.session_state['selected_color']

        # Ensure CMYK bar styles are available on every rerun
        st.markdown(
            """
<style>
.pp-cmyk-wrap{font-size:0.85rem;line-height:1.25;margin-top:0.25rem;}
.pp-cmyk-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:0.4rem;}
.pp-badge{padding:4px 8px;border-radius:6px;font-weight:600;display:inline-flex;align-items:center;gap:4px;box-shadow:0 1px 2px rgba(0,0,0,0.15);} 
.pp-c{background:#00bcd4;color:#00363d;}
.pp-m{background:#e91e63;color:#fff;}
.pp-y{background:#ffeb3b;color:#3a3300;}
.pp-k{background:#424242;color:#fff;}
.pp-bar-block{margin:2px 0 6px 0;}
.pp-bar{height:14px;background:#eee;border-radius:7px;overflow:hidden;position:relative;margin-top:12px}
.pp-fill{height:100%;position:absolute;left:0;top:0;}
.pp-fill.c{background:#00bcd4;}
.pp-fill.m{background:#e91e63;}
.pp-fill.y{background:#ffd600;}
.pp-fill.k{background:#424242;}
.pp-bar-label{display:flex;justify-content:space-between;font-size:0.70rem;margin-top:2px;font-weight:600;}
@media (prefers-color-scheme: dark){
    .pp-bar{background:#333;}
    .pp-c{color:#002b30;}
    .pp-y{color:#332b00;}
}
</style>
""",
            unsafe_allow_html=True,
        )

        st.markdown("### 🧪 Color Explorer (Click a Swatch)")
        swatch_size = 40  # compact
        # Ordering logic (note: CSV indices remain stable, we only change display order)
        def _compute_mix_path(cdata):
            # Build a path attempting monotonic non-decreasing CMYK progression (add-only mixing)
            remaining = cdata[:]
            path = []
            resets = 0
            # Helper to extract totals
            def total(cd):
                return cd['C%'] + cd['M%'] + cd['Y%'] + cd['K%']
            # Start with lowest total ink load
            current = min(remaining, key=total) if remaining else None
            while remaining:
                if current is None:
                    break
                if current in remaining:
                    remaining.remove(current)
                    path.append(current)
                if not remaining:
                    break
                # Candidates with all channels >= current
                if current is not None:
                    candidates = [cd for cd in remaining if all(cd[ch+'%'] >= current[ch+'%'] for ch in ['C','M','Y','K'])]
                else:
                    candidates = []
                if current is not None and candidates:
                    # Choose minimal total increase; tiebreak by smallest max single-channel jump then by frequency (Count desc)
                    cur = current  # capture
                    def score(cd):
                        inc = total(cd) - total(cur)
                        max_jump = max(cd[ch+'%'] - cur[ch+'%'] for ch in ['C','M','Y','K'])
                        return (inc, max_jump, -cd['Count'])
                    current = min(candidates, key=score)
                else:
                    # Reset: pick lowest total among remaining to start a new additive chain
                    resets += 1
                    current = min(remaining, key=total)
            # Annotate resets in path entries (optional)
            if resets > 0:
                # Mark boundaries so user can see where a fresh mix would be needed
                last_vals = None
                chain_id = 0
                for cd in path:
                    vals = tuple(cd[ch+'%'] for ch in ['C','M','Y','K'])
                    if last_vals is not None and any(v < lv for v,lv in zip(vals,last_vals)):
                        chain_id += 1
                    cd['MixChain'] = chain_id
                    last_vals = vals
            else:
                for cd in path:
                    cd['MixChain'] = 0
            return path

        if order_choice == 'Mix Path':
            color_data = _compute_mix_path(color_data)
        elif order_choice == 'Count (desc)':
            color_data.sort(key=lambda d: -d['Count'])
        elif order_choice == 'Hue':
            import colorsys
            def _h(cd):
                r,g,b = [int(x) for x in cd['RGB'][4:-1].split(',')]
                return colorsys.rgb_to_hsv(r/255.0,g/255.0,b/255.0)[0]
            color_data.sort(key=_h)
        elif order_choice == 'Luminance':
            color_data.sort(key=lambda cd: sum(int(x) for x in cd['RGB'][4:-1].split(',')))
        elif order_choice == 'Cyan %':
            color_data.sort(key=lambda cd: -cd['C%'])
        elif order_choice == 'Magenta %':
            color_data.sort(key=lambda cd: -cd['M%'])
        elif order_choice == 'Yellow %':
            color_data.sort(key=lambda cd: -cd['Y%'])
        elif order_choice == 'Black %':
            color_data.sort(key=lambda cd: -cd['K%'])
        elif order_choice == 'Index':
            color_data.sort(key=lambda cd: cd['Index'])

        # Toggle to show names on swatches (may reduce grid density)
        show_names = st.checkbox("Show names on swatches", value=False, help="Overlay color names (truncated) on swatch buttons")

        # Precompute hue/luminance for future ordering without recomputation (cached in session)
        if 'pp_color_metrics' not in st.session_state:
            st.session_state['pp_color_metrics'] = {}
        metrics = st.session_state['pp_color_metrics']
        import colorsys
        for cd in color_data:
            idx = cd['Index']
            if idx not in metrics:
                r,g,b = [int(x) for x in cd['RGB'][4:-1].split(',')]
                h,s,v = colorsys.rgb_to_hsv(r/255.0,g/255.0,b/255.0)
                lum = 0.299*r + 0.587*g + 0.114*b
                metrics[idx] = {'h':h,'lum':lum}

        # Grid of swatches: we render a solid color image then a button below so background color doesn't rely on fragile DOM hooks
        num_cols = 16 if len(color_data) >= 16 else max(1, len(color_data))
        for row_start in range(0, len(color_data), num_cols):
            cols = st.columns(num_cols, gap="small")
            for ci, col in enumerate(cols):
                pos = row_start + ci
                if pos >= len(color_data):
                    continue
                entry = color_data[pos]
                r,g,b = [int(x) for x in entry['RGB'][4:-1].split(',')]
                sw_img = Image.new('RGB',(swatch_size,swatch_size),(r,g,b))
                with col:
                    st.image(sw_img, caption=None, use_container_width=False)
                    is_sel = (st.session_state.get('selected_color') == entry['Index'])
                    base_label = f"#{entry['Index']}" + (f" {entry['Color Name'][:6]}" if show_names else "")
                    label = base_label + (" ✓" if is_sel else "")
                    if st.button(label, key=f"pp_swatch_btn_{entry['Index']}"):
                        st.session_state['selected_color'] = entry['Index']
                        st.rerun()

        # Selected color detail panel (lookup by original Index in base map)
        if selected_idx is not None:
            sel = base_by_index.get(selected_idx)
        else:
            sel = None
        if sel is not None:
            # Compact heading with reduced top margin
            st.markdown(
                f"""
<style>
.pp-selected-color h3 {{ margin-top:0.6rem !important; margin-bottom:0.6rem !important; }}
</style>
<div class='pp-selected-color'>
<h3>🎯 Selected Color: Index {sel['Index']} – {sel['Color Name']}</h3>
</div>
""",
                unsafe_allow_html=True,
            )
            sw_img = Image.new('RGB', (150,150), tuple(int(x) for x in sel['RGB'][4:-1].split(',')))
            dc1, dc2 = st.columns([1,2])
            with dc1:
                st.image(sw_img, caption='Swatch', width=150)
                if st.button('Clear Selection', key='pp_clear_sel'):
                    st.session_state['selected_color'] = None
                    st.rerun()
            with dc2:
                total_px = results['processed_image'].size[0] * results['processed_image'].size[1]
                # Emphasized CMYK-centric presentation
                c_pct, m_pct, y_pct, k_pct = sel['C%'], sel['M%'], sel['Y%'], sel['K%']
                usage_fraction = sel['Count'] / total_px
                pixel_count = sel['Count']
                st.markdown(
                    f"""
<div class='pp-cmyk-wrap'>
    <div class='pp-cmyk-row'>
        <span class='pp-badge pp-c'>C {c_pct}%</span>
        <span class='pp-badge pp-m'>M {m_pct}%</span>
        <span class='pp-badge pp-y'>Y {y_pct}%</span>
        <span class='pp-badge pp-k'>K {k_pct}%</span>
    </div>
    <div class='pp-bar-block'>
        <div class='pp-bar'><div class='pp-fill c' style='width:{c_pct}%;'></div></div>
        <div class='pp-bar-label'><span>Cyan</span><span>{c_pct}%</span></div>
        <div class='pp-bar'><div class='pp-fill m' style='width:{m_pct}%;'></div></div>
        <div class='pp-bar-label'><span>Magenta</span><span>{m_pct}%</span></div>
        <div class='pp-bar'><div class='pp-fill y' style='width:{y_pct}%;'></div></div>
        <div class='pp-bar-label'><span>Yellow</span><span>{y_pct}%</span></div>
        <div class='pp-bar'><div class='pp-fill k' style='width:{k_pct}%;'></div></div>
        <div class='pp-bar-label'><span>Black</span><span>{k_pct}%</span></div>
    </div>
    <div><strong>RGB:</strong> {sel['RGB']}<br>
         <strong>CMYK (0-1):</strong> C {sel['C']}, M {sel['M']}, Y {sel['Y']}, K {sel['K']}<br>
         <strong>Parts (CMY):</strong> {sel['CMY Parts']}<br>
         <strong>Pixel Count:</strong> {pixel_count:,} pixels ({usage_fraction:.2%} of image)
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

        # Downloads
        st.subheader("⬇️ Downloads")
        # Build ordered list of available downloads
        btn_specs = []
        if 'cmyk_data' in results['files']:
            btn_specs.append(("📊 Color CSV", 'cmyk_data', 'cmyk_color_data.csv', 'text/csv'))
        if 'mixing_data' in results['files']:
            btn_specs.append(("🧪 Mixing CSV", 'mixing_data', 'cmyk_mixing_recipes.csv', 'text/csv'))
        if 'visualization' in results['files'] and results.get('visualization_type')=='png':
            btn_specs.append(("🖼️ Viz PNG", 'visualization', 'color_visualization.png', 'image/png'))
        if 'reconstructed' in results['files']:
            btn_specs.append(("🔄 Reconstructed", 'reconstructed', 'reconstructed_image.png', 'image/png'))
        
        # Add print swatch sheet generation
        print_col1, print_col2, print_col3, print_col4 = st.columns([1, 1, 1, 1])
        with print_col1:
            page_size = st.selectbox("Print Page Size", ["a4", "letter"], index=0, help="Choose paper size for swatch sheet")
        with print_col2:
            swatches_per_row = st.selectbox("Swatches Per Row", [4, 6, 8, 10, 12, 16], index=2, help="Number of color swatches per row")
        with print_col3:
            output_format = st.selectbox("Output Format", ["png", "pdf"], index=0, help="Choose PNG (image) or PDF format")
        with print_col4:
            square_size_inches = st.selectbox("Square Size", [0.15, 0.2, 0.25, 0.3, 0.35, 0.4], index=1, 
                                            format_func=lambda x: f'{x}"', 
                                            help="Size of color squares in inches")
        
        # Additional option for visualization inclusion
        include_visualization = st.checkbox("Include visualization on print sheet", value=False, 
                                          help="Add the color visualization to the printed swatch sheet (uses more space)")

        if st.button("🖨️ Generate Print Swatch Sheet", help="Create a printable sheet with all color swatches for physical comparison"):
            with st.spinner("Generating print swatch sheet..."):
                try:
                    if output_format == "pdf":
                        swatch_data = create_print_swatch_sheet(color_data, results, page_size, swatches_per_row, "pdf", square_size_inches, include_visualization)
                        if swatch_data is not None:
                            # Add to files for ZIP download
                            results['files']['print_swatches_pdf'] = swatch_data
                            
                            # Open PDF in new tab instead of downloading
                            pdf_b64 = base64.b64encode(swatch_data).decode('utf-8')
                            pdf_data_url = f"data:application/pdf;base64,{pdf_b64}"
                            
                            # JavaScript to open PDF in new tab
                            st.markdown(f"""
                            <script>
                            window.open('{pdf_data_url}', '_blank');
                            </script>
                            """, unsafe_allow_html=True)
                            
                            # Show success message
                            st.success("✅ PDF swatch sheet generated and opened in new tab!")
                            st.info("💡 If the PDF didn't open, please allow popups for this site. You can also download it below.")
                            
                            # Clear ZIP cache since we added a new file
                            st.session_state.pop('pp_cached_zip', None)
                    else:
                        swatch_sheet = create_print_swatch_sheet(color_data, results, page_size, swatches_per_row, "png", square_size_inches, include_visualization)
                        
                        # Convert to bytes for download
                        img_buffer = BytesIO()
                        swatch_sheet.save(img_buffer, format='PNG', dpi=(300, 300))
                        swatch_bytes = img_buffer.getvalue()
                        
                        # Add to files for ZIP download
                        results['files']['print_swatches'] = swatch_bytes
                        
                        # Show preview
                        viz_note = " (with viz)" if include_visualization else ""
                        st.image(swatch_sheet, caption=f"Print Swatch Sheet ({page_size.upper()}, {swatches_per_row} per row, {square_size_inches}\" squares{viz_note})", width=400)
                        st.success("✅ PNG swatch sheet generated! Available in downloads below.")
                        
                        # Clear ZIP cache since we added a new file
                        st.session_state.pop('pp_cached_zip', None)
                    
                except Exception as e:
                    st.error(f"Error generating swatch sheet: {e}")

        # Add print swatch files to download buttons if they exist
        if 'print_swatches' in results['files']:
            btn_specs.append(("🖨️ Print PNG", 'print_swatches', 'color_swatches_print.png', 'image/png'))
        if 'print_swatches_pdf' in results['files']:
            btn_specs.append(("📄 Print PDF", 'print_swatches_pdf', 'color_swatches_print.pdf', 'application/pdf'))

        # Create custom inline download buttons (anchors) without spacing
        html_parts = [
            """
<style>
.pp-dl-row{display:flex;flex-wrap:wrap;}
/* Match earlier Streamlit compact sizing (~0.75-0.8rem) while keeping buttons flush */
.pp-dl-row .pp-btn{margin:0;display:inline-flex;align-items:center;justify-content:center;gap:4px;padding:6px 14px;font-size:0.78rem;line-height:1.1;font-weight:600;background:#444;color:#fff;border:1px solid #222;border-right:none;border-radius:0;text-decoration:none;min-height:34px;}
.pp-dl-row .pp-btn:first-child{border-top-left-radius:6px;border-bottom-left-radius:6px;}
.pp-dl-row .pp-btn:last-child{border-right:1px solid #222;border-top-right-radius:6px;border-bottom-right-radius:6px;}
.pp-dl-row .pp-btn:hover{background:#555;}
.pp-dl-row .pp-btn:active{background:#666;}
.pp-dl-row .pp-btn:focus{outline:2px solid #ffbf00;outline-offset:1px;}
@media (max-width:700px){.pp-dl-row{flex-direction:column;}.pp-dl-row .pp-btn{border-radius:6px !important;border-right:1px solid #222 !important;}}
</style>
<div class='pp-dl-row'>
"""
        ]
        for label, key_in_files, filename, mime in btn_specs:
            if key_in_files is None:
                # Cache the zip in session state to avoid recomputation on every rerender
                cache_key = 'pp_cached_zip'
                if (cache_key not in st.session_state or
                        st.session_state.get('pp_cached_zip_viz_type') != results.get('visualization_type')):
                    st.session_state[cache_key] = create_download_zip(
                        results['files'], visualization_type=results.get('visualization_type')
                    )
                    st.session_state['pp_cached_zip_viz_type'] = results.get('visualization_type')
                data_bytes = st.session_state[cache_key]
            else:
                data_bytes = results['files'][key_in_files]
            b64 = base64.b64encode(data_bytes).decode('utf-8')
            html_parts.append(
                f"<a class='pp-btn' download='{filename}' href='data:{mime};base64,{b64}'>{label}</a>"
            )
        html_parts.append("</div>")
        st.markdown(''.join(html_parts), unsafe_allow_html=True)

    if uploaded_file is not None:
        # Show original and sidebar params
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Size: {image.size[0]}x{image.size[1]}", width=300)
        with col2:
            st.subheader("ℹ️ Upload Info")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File Size:** {len(uploaded_file.getvalue())} bytes")
            st.write(f"**Image Dimensions:** {image.size[0]} x {image.size[1]} pixels")

        # Clear cached results if new file
        if st.session_state.get('uploaded_filename') != uploaded_file.name:
            st.session_state['uploaded_filename'] = uploaded_file.name
            st.session_state.pop('pp_results', None)
            st.session_state['selected_color'] = None  # reset selection on new image

        st.sidebar.subheader("🔍 Scaling Options")
        scale_mode = st.sidebar.selectbox("Scaling Mode", ["Pixel Dimensions","Original Size","Percentage"], help="Choose how to scale the image before processing")
        scale_value1 = scale_value2 = None
        if scale_mode == "Percentage":
            scale_value1 = st.sidebar.number_input("Scale Percentage", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
        elif scale_mode == "Pixel Dimensions":
            col_w, col_h = st.sidebar.columns(2)
            with col_w: scale_value1 = st.sidebar.number_input("Width", min_value=16, max_value=2048, value=32)
            with col_h: scale_value2 = st.sidebar.number_input("Height", min_value=16, max_value=2048, value=32)
        resample_method = st.sidebar.selectbox("Resampling Method", ["nearest","bilinear","bicubic","lanczos"], index=0)

        st.sidebar.subheader("🎨 Color Reduction")
        approx_method = st.sidebar.selectbox("Color Reduction Method", ["precision","kmeans","median_cut"], index=1)
        if approx_method == 'precision':
            precision = st.sidebar.slider("Precision", 0.01, 0.2, 0.05, step=0.01)
            n_colors = None
        else:
            precision = None
            n_colors = st.sidebar.slider("Number of Colors", 8, 64, 16)

        st.sidebar.subheader("📊 Output Options")
        visualize_type = st.sidebar.selectbox("Visualization Type", ["simple","html","none"], index=0)
        reconstruct = st.sidebar.checkbox("Generate Reconstructed Image", value=False)
        save_cmyk = st.sidebar.checkbox("Save Full CMYK Pixel Data", value=False)

        # Single processing trigger
        trigger_processing = False
        if sample_default and 'pp_results' not in st.session_state:
            trigger_processing = True
        if st.sidebar.button("🚀 Process Image", type="primary"):
            trigger_processing = True

        if trigger_processing:
            with st.spinner("Processing image... This may take a moment."):
                try:
                    results = process_image_streamlit(
                        uploaded_file, scale_mode, scale_value1, scale_value2,
                        resample_method, approx_method, precision, n_colors,
                        visualize_type, save_cmyk, reconstruct
                    )
                    st.session_state['pp_results'] = results
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    st.exception(e)

        if 'pp_results' in st.session_state:
            prev = st.session_state['pp_results']
            build_and_render(prev, prev.get('approx_method', approx_method), reconstruct, visualize_type)
    else:
        st.info("👆 Please upload an image file to get started!")
        st.markdown("""## 📖 How to Use

1. **Upload an Image** - Choose a PNG, JPG, or JPEG file
2. **Configure Parameters** - Adjust scaling, color reduction, and output options in the sidebar
3. **Process** - Click the "Process Image" button
4. **Download Results** - Get your color data, visualizations, and reconstructed images

## 🎯 Parameter Guide

- **Scaling**: Control image size for processing (smaller = faster, fewer colors)
- **Color Reduction Method**:
  - `precision`: General purpose
  - `kmeans`: Paint-by-number style
  - `median_cut`: Preserves key colors
- **Visualization**:
  - `simple`: PNG grid with indices
  - `html`: Interactive web page
  - `none`: Skip visualization
""")

if __name__ == "__main__":
    main()
