# EIT Reconstruction Dashboard

**Interactive Streamlit dashboard for analyzing Electrical Impedance Tomography (EIT) reconstruction methods.**

## 📊 Features

### Five Interactive Views

1. **🏆 Leaderboard** - Dynamic rankings with customizable composite scores
2. **📉 Degradation Curve** - Performance trends across different samples
3. **🔍 Side-by-Side Comparison** - Compare any two methods on any sample
4. **⚠️ Failure Gallery** - Analyze worst-performing samples per method
5. **📡 Radar Chart** - Multi-dimensional metric visualization

### Interactive Composite Weight Editor

The dashboard's **key insight feature**: five sliders in the sidebar that let you rebalance the composite score formula in real-time:

- **Tier 1**: KTC Score (Primary benchmark - lower is better)
- **Tier 2**: Dice Coefficients (Overlap metrics for resistive/conductive regions)
- **Tier 3**: IoU Scores (Intersection over Union metrics)
- **Tier 4**: Hausdorff Distance (Boundary accuracy - 95th percentile)
- **Tier 5**: Overall Balance (Balancing factor for performance)

**When you adjust the sliders, the leaderboard rankings update instantly**, showing how different metric priorities change the method rankings.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit plotly pandas matplotlib pillow seaborn --break-system-packages
```

### 2. Ensure Data Files Exist

The dashboard requires two JSON files:

- `scores.json` - Averaged metrics for each method
- `outputs/per_run_metrics.json` - Per-sample metrics for each method

These are automatically generated when you run `example_usage.py`.

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 4. Test the Dashboard

Verify all functions work correctly:

```bash
python test_dashboard.py
```

## 📁 Data Structure

### scores.json Format

```json
{
    "Method Name": {
        "Dice (resistive)": 0.7234,
        "Dice (conductive)": 0.6891,
        "IoU (resistive)": 0.5678,
        "IoU (conductive)": 0.5234,
        "KTC score": 0.1523
    }
}
```

### per_run_metrics.json Format

```json
{
  "method_key": {
    "sample_id": {
      "ktc_score": 0.1234,
      "dice_resistive": 0.7456,
      "dice_conductive": 0.7123,
      "iou_resistive": 0.5943,
      "iou_conductive": 0.5534,
      "hd95_resistive": 12.34,
      "hd95_conductive": 15.67
    }
  }
}
```

## 🎯 View Details

### 1. Leaderboard View

- **Interactive bar chart** showing composite scores for all methods
- **Color-coded by grade**: 🟢 A (≥85), 🔵 B (≥70), 🟡 C (≥55), 🔴 D (<55)
- **Detailed metrics table** with all performance indicators
- **Real-time updates** when you adjust weight sliders

**Weight Controls:**
- Adjust five sliders to change metric priorities
- Click **"Normalize Weights"** to ensure they sum to 1.0
- Click **"Reset to Defaults"** to restore original weights
- Visual progress bars show current weight distribution

### 2. Degradation Curve View

- **Multi-method line chart** showing KTC score trends across samples
- **Method selector** to choose which methods to display
- **Statistics table** with mean, std dev, min, max, and range for each method
- Hover over data points for detailed sample information

### 3. Side-by-Side Comparison View

- **Method selectors** to choose any two methods
- **Sample selector** to choose which sample to compare
- **Metrics comparison table** showing differences between methods
- **Radar chart** visualizing performance across five key metrics
- **Visual comparison** (when images available in `outputs/comparison_panels/`)

### 4. Failure Gallery View

- **Worst 3 samples** for each method (highest KTC scores)
- **Metric cards** showing detailed performance for each failure case
- **Visual thumbnails** (when available)
- Helps identify systematic failure patterns

### 5. Radar Chart View

- **Multi-method overlay** on a single radar chart
- **Metric selector** to choose which dimensions to compare
- **Statistics table** showing mean, std dev, min, max for each metric
- Excellent for understanding method trade-offs

## 🔧 Customization

### Modifying Composite Score Formula

Edit the `calculate_composite_score()` function in `app.py` to change how metrics are combined:

```python
def calculate_composite_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    # Extract metrics
    ktc = metrics.get('KTC score', 0)
    dice_r = metrics.get('Dice (resistive)', 0)
    # ... add your custom logic here
    
    # Calculate weighted score
    composite = (
        weights['tier1'] * tier1_score +
        weights['tier2'] * tier2_score +
        # ... your custom tiers
    ) / sum(weights.values())
    
    return composite
```

### Adding New Metrics

1. Update the `scores.json` and `per_run_metrics.json` files with new metrics
2. Add new tiers to the composite score calculation
3. Add new sliders in the sidebar
4. Update the radar chart metric selector

### Changing Color Scheme

Modify the `COLORS` dictionary in `app.py`:

```python
COLORS = {
    'water': '#1a3a5c',
    'resistive': '#D85A30',
    'conductive': '#1D9E75',
    'primary': '#1a3a5c',
    'success': '#1D9E75',
    'warning': '#F5A623',
    'danger': '#D85A30'
}
```

## 📈 Example Workflow

### Scenario: Comparing reconstruction methods

1. **Run the benchmark**: `python example_usage.py` to generate data
2. **Launch dashboard**: `streamlit run app.py`
3. **Review leaderboard** to see overall rankings
4. **Adjust weights** - Try emphasizing different metrics:
   - Move Tier 1 (KTC) slider to 0.6 → See how pure accuracy ranking changes
   - Move Tier 2 (Dice) slider to 0.5 → Emphasize overlap metrics
   - Move Tier 4 (HD95) slider to 0.4 → Prioritize boundary accuracy
5. **Check degradation curve** to see which methods degrade gracefully
6. **Compare top two methods** side-by-side on worst sample
7. **Analyze failures** to understand systematic issues
8. **Use radar chart** to understand trade-offs between metrics

## 🎨 Visual Examples

### Leaderboard with Weight Editor
```
Sidebar:
┌─────────────────────────────┐
│ ⚙️ Composite Score Weights  │
├─────────────────────────────┤
│ Tier 1: KTC Score     [====]│ 0.40
│ Tier 2: Dice Coeff.   [===] │ 0.25
│ Tier 3: IoU Scores    [==]  │ 0.20
│ Tier 4: Hausdorff     [=]   │ 0.10
│ Tier 5: Balance       [=]   │ 0.05
├─────────────────────────────┤
│ [⚖️ Normalize Weights]      │
│ [🔄 Reset to Defaults]      │
└─────────────────────────────┘

Main View:
┌─────────────────────────────┐
│ Method Rankings             │
├─────────────────────────────┤
│ 1. Gauss-Newton    84.1 (B) │ ████████████████████
│ 2. Back-projection 76.7 (B) │ ████████████████
│ 3. Mock baseline   10.6 (D) │ ██
└─────────────────────────────┘
```

### Real-Time Update Example

**Before** (Default weights: KTC=0.4, Dice=0.25, IoU=0.2, HD95=0.1, Balance=0.05):
```
1. Gauss-Newton    84.1 (B)
2. Back-projection 76.7 (B)
3. Mock baseline   10.6 (D)
```

**After** (Adjusted: KTC=0.6, Dice=0.15, IoU=0.15, HD95=0.05, Balance=0.05):
```
1. Gauss-Newton    87.3 (A) ← Grade improved!
2. Back-projection 78.9 (B) ← Score improved!
3. Mock baseline    8.2 (D) ← Score decreased!
```

Rankings update **instantly** when you move any slider!

## 🐛 Troubleshooting

### Dashboard won't load
```bash
# Check if data files exist
ls scores.json outputs/per_run_metrics.json

# If missing, run the benchmark first
python example_usage.py
```

### Import errors
```bash
# Reinstall dependencies
pip install streamlit plotly pandas matplotlib pillow seaborn --break-system-packages
```

### Port already in use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Empty visualizations
- Ensure `outputs/comparison_panels/` directory exists
- Check that PNG files are named correctly: `method_sample_X.png`

## 📝 Technical Details

### Composite Score Calculation

The composite score (0-100) is calculated as:

```
composite = (w₁·T₁ + w₂·T₂ + w₃·T₃ + w₄·T₄ + w₅·T₅) / Σwᵢ

where:
  T₁ = (1 - KTC) × 100          (KTC inverted since lower is better)
  T₂ = mean(Dice_R, Dice_C) × 100
  T₃ = mean(IoU_R, IoU_C) × 100
  T₄ = mean(1 - HD95_R/100, 1 - HD95_C/100) × 100
  T₅ = T₁                        (Balance tier)
  
  wᵢ = user-defined weights from sliders
```

### Grade Assignment

```python
A: score ≥ 85  (Excellent)
B: score ≥ 70  (Good)
C: score ≥ 55  (Acceptable)
D: score < 55  (Poor)
```

## 🎓 Educational Use

This dashboard is designed for:

- **Research presentations** - Interactive demos of method comparisons
- **Method development** - Quick feedback on new reconstruction algorithms
- **Student education** - Visual understanding of EIT metrics and trade-offs
- **Benchmark analysis** - Systematic comparison across large method sets

## 📄 License

Same license as the ktc-eit-framework project.

## 🙏 Acknowledgments

Built for the KTC EIT Challenge framework. Uses real data from the official KTC training set.

---

**Friday Output**: ✅ Complete dashboard with all five views functional, weight sliders updating rankings in real-time, and tested end-to-end with real data structure.
