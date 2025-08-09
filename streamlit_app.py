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
         <strong>Usage Fraction:</strong> {usage_fraction:.2%}
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
        if results['files']:
            btn_specs.append(("📦 All (ZIP)", None, 'painting_parser_results.zip', 'application/zip'))

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
