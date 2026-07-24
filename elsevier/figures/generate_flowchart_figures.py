import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

output_dir = '/Users/saranboddu/Desktop/Amrita/Extra/RhizoWhisperer/RHIZO-NET: Root Health and Integrated Zonal Optimization Network via Edaphic Topology/elsevier/figures'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def create_master_flowchart():
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('#f8f9fa')
    
    # Title
    ax.text(0.5, 0.97, 'RHIZO-NET Master End-to-End Pipeline Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#1a252f')
    ax.text(0.5, 0.945, 'Multi-Modal Ingestion → ONNX Neural Segmentation → Uncertainty Routing → Topology & Edaphic Fusion → Agronomic & Climate Deployment', 
            ha='center', va='center', fontsize=11, fontstyle='italic', color='#5c6b73')

    # Color palette
    c_in = '#e1f5fe'      # Light Blue
    c_seg = '#e8f5e9'     # Light Green
    c_rout = '#fff3e0'    # Light Orange
    c_skel = '#f3e5f5'    # Light Purple
    c_fusion = '#fce4ec'  # Light Pink
    c_out = '#e0f2f1'     # Light Teal

    # Helper function to draw box
    def draw_box(x, y, w, h, text, color='#ffffff', border='#2c3e50', title=None):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", 
                                      linewidth=1.5, edgecolor=border, facecolor=color)
        ax.add_patch(rect)
        if title:
            ax.text(x + w/2, y + h - 0.03, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
            ax.text(x + w/2, y + (h - 0.03)/2, text, ha='center', va='center', fontsize=9, color='#34495e', multialignment='center')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold', color='#2c3e50', multialignment='center')

    # Helper function to draw arrow
    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2, mutation_scale=15))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.015, label, ha='center', va='center', fontsize=8, fontweight='bold', color='#e74c3c')

    # Column 1: Multi-Modal Inputs
    draw_box(0.04, 0.75, 0.18, 0.15, "106,900 Images\nacross 6 Datasets\n(RootNav, PRMI, etc.)", c_in, border='#0288d1', title="Visual Input")
    draw_box(0.04, 0.52, 0.18, 0.15, "ISRIC SoilGrids 2.0\n0-200 cm Profiles\n(pH, SOC, N, CEC, Clay)", c_in, border='#0288d1', title="Edaphic Input")
    draw_box(0.04, 0.29, 0.18, 0.15, "IPCC Scenarios\nRCP 4.5 & RCP 8.5\n(Temp & Evapotranspiration)", c_in, border='#0288d1', title="Climate Input")

    # Column 2: Deep Learning ONNX Segmentation Suite
    draw_box(0.27, 0.25, 0.23, 0.67, 
             "1. RhizoUNet (1.7M params, 4.2ms, 94.2% IoU)\n   • ELU Activations + Average Pooling\n\n"
             "2. DualStreamRootNet (4.8M params, 9.6ms, 96.5% IoU)\n   • Frangi Hessian Vesselness Stream\n\n"
             "3. RhizoHybridTransformer (79.7K params, 1.8ms)\n   • Swin Shifted Window + Root Query Tokens\n\n"
             "4. RhizoAttentionNet (5.8M params, 12.8ms, 97.9% IoU)\n   • OTAM Directional Filters + MSRFP", 
             c_seg, border='#2e7d32', title="Stage 2: ONNX Model Suite")

    # Arrows Input -> Segmentation
    draw_arrow(0.22, 0.825, 0.27, 0.825)

    # Column 3: Uncertainty Routing & GRSR Gap Repair
    draw_box(0.54, 0.65, 0.18, 0.25, 
             "Entropy Check (H > 0.45)\nConfidence < 0.50?\n\n↓ Yes: MobileSAM Adapter\n↓ No: Direct Soft Mask", 
             c_rout, border='#ef6c00', title="Stage 3: MobileSAM Routing")
    
    draw_box(0.54, 0.29, 0.18, 0.28, 
             "Distance Transform\nOpen Endpoint Detection\n\nGeodesic Path Propagation\n(Restores 100% Occluded Gaps)", 
             c_skel, border='#8e24aa', title="Stage 4: GRSR Gap Repair")

    # Arrows Seg -> Routing -> GRSR
    draw_arrow(0.50, 0.775, 0.54, 0.775)
    draw_arrow(0.63, 0.65, 0.63, 0.57)

    # Column 4: Topology, Graph Encoding & Edaphic Fusion
    draw_box(0.76, 0.65, 0.20, 0.25, 
             "skan Graph Extraction\nSholl Branching Analysis\nSeminal Root Angle Map", 
             c_skel, border='#8e24aa', title="Stage 5: skan Morphometrics")

    draw_box(0.76, 0.45, 0.20, 0.16, 
             "RhizoGraphFormer\nLaplacian Positional Encoding\n128-d Topological Vector", 
             c_fusion, border='#c2185b', title="Stage 6: Graph Encoding")

    draw_box(0.76, 0.25, 0.20, 0.16, 
             "RhizoFusionNet (PyG GNN)\n128-d Vector + 30-d SoilGrids\n5-Class Deficiency Diagnosis", 
             c_fusion, border='#c2185b', title="Stage 7: Edaphic Fusion")

    # Arrows GRSR -> skan -> GraphFormer -> Fusion
    draw_arrow(0.72, 0.43, 0.76, 0.775)
    draw_arrow(0.86, 0.65, 0.86, 0.61)
    draw_arrow(0.86, 0.45, 0.86, 0.41)
    draw_arrow(0.22, 0.595, 0.76, 0.33, label="Soil Profiles")

    # Bottom Row: Actionable Deployment Outputs
    draw_box(0.04, 0.04, 0.28, 0.16, 
             "TNAU Agronomic Engine\n5 Crop Prescriptions (NPK Splits,\nGypsum, Foliar Spray Cards)", 
             c_out, border='#00695c', title="Stage 8: Agronomic Engine")

    draw_box(0.36, 0.04, 0.28, 0.16, 
             "CARRS Climate Simulator\nDrought Resilience Index (DRI)\nunder RCP 4.5 vs RCP 8.5", 
             c_out, border='#00695c', title="Stage 9: CARRS Simulator")

    draw_box(0.68, 0.04, 0.28, 0.16, 
             "RCS-Flux Carbon Predictor\n1.76 t CO2e/ha/yr Sequestration\n$35.20/ha/yr ROI Valuation", 
             c_out, border='#00695c', title="Stage 10: Carbon Economics")

    # Arrows to Bottom Row
    draw_arrow(0.86, 0.25, 0.18, 0.20)
    draw_arrow(0.86, 0.65, 0.50, 0.20)
    draw_arrow(0.86, 0.65, 0.82, 0.20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_master_flowchart.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated fig_master_flowchart.png")

create_master_flowchart()
