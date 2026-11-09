# FloodSight

# FloodSight – Satellite-Based Flood Intelligence Platform

**Turning satellite imagery into meaningful flood insights.**

Floods are among the most destructive natural disasters, affecting communities, infrastructure, agriculture, and livelihoods. Understanding where flooding occurs and who or what may be exposed is essential for effective disaster response and planning.

**FloodSight** is a geospatial flood intelligence platform that uses satellite imagery and deep learning to identify potential flood-affected areas and explore their possible impacts. It brings together image segmentation, geospatial analysis, uncertainty estimation, and interactive visualization in one application.

The goal is to make satellite-based flood analysis more accessible, interpretable, and useful for people interested in disaster monitoring, environmental analysis, and geospatial research.

## Live Application

Explore the deployed application:

**[Open FloodSight](https://floodsight-3.streamlit.app/)**

Source code:

**[GitHub Repository](https://github.com/zobayerzo/floodsight)**

## What can FloodSight do?

FloodSight combines machine learning predictions with geospatial information to provide a broader picture of potential flood impacts.

### 1. Deep Learning-Based Flood Detection

The platform uses a U-Net-based semantic segmentation model to identify water and potential flood-affected regions from satellite imagery.

Rather than treating satellite images as ordinary photographs, the model analyzes their spatial patterns to distinguish water-related areas from surrounding land.

### 2. Interactive Flood Visualization

Flood predictions can be explored through interactive geospatial maps. This helps users understand the spatial distribution of detected water areas and examine the results in their geographic context.

### 3. Prediction Uncertainty Analysis

FloodSight incorporates Monte Carlo Dropout (MC Dropout) to estimate prediction uncertainty.

This provides an additional perspective on model predictions by highlighting areas where the model produces less consistent results. It is particularly useful when interpreting predictions in complex geographical conditions.

### 4. Population Exposure Analysis

FloodSight combines predicted flood areas with population-related geospatial information to estimate potential population exposure.

This analysis provides an indication of how many people may be located within or near predicted water-affected areas. It should be interpreted as a spatial exposure estimate rather than a verified count of affected individuals.

### 5. Agricultural Exposure Analysis

Flooding can have serious consequences for agricultural land and food production. FloodSight includes agricultural exposure analysis to explore the potential overlap between predicted water areas and agricultural land.

### 6. Geospatial Data Export

The platform supports exporting analytical results for further examination and use in other geospatial workflows.

Supported export formats include GeoTIFF and CSV.

## How It Works

FloodSight follows a workflow that connects satellite imagery, deep learning, and geospatial analysis.

1. **Satellite imagery:** Input imagery provides the geographical information used for flood analysis.
2. **Image preprocessing:** The imagery is prepared for model inference.
3. **Flood segmentation:** A trained U-Net model generates pixel-level predictions.
4. **Uncertainty estimation:** MC Dropout provides an estimate of prediction uncertainty.
5. **Geospatial analysis:** Predictions are combined with relevant geographical datasets to explore population and agricultural exposure.
6. **Visualization and export:** Results are presented through interactive maps and can be exported for additional analysis.

## Model and Dataset

FloodSight uses the **Sen1Floods11 v1.1** dataset for model development and evaluation.

Sen1Floods11 is a satellite-based flood segmentation dataset containing imagery and corresponding flood labels from multiple geographical regions and flood events.

The project dataset includes:

- 4,831 image chips.
- 446 hand-labelled image chips.
- Data covering 11 geographical regions.

The model was evaluated using the official test samples available for the project.

### Model Evaluation

The following results were obtained on the project's test evaluation:

| Evaluation Metric | Score |
|---|---:|
| Intersection over Union (IoU) | 0.6629 |
| Dice Coefficient | 0.7972 |
| Precision | 0.8094 |
| Recall | 0.7855 |
| Accuracy | 0.9509 |

These results provide an indication of the model's performance on the evaluated test samples. They should not be interpreted as a guarantee of equivalent performance across all geographical regions, satellite conditions, or flood events.

## Technology Stack

FloodSight brings together tools from machine learning, scientific computing, and geospatial development.

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Application Framework | Streamlit |
| Deep Learning | PyTorch |
| Segmentation Architecture | U-Net |
| Geospatial Processing | Rasterio, GeoPandas, Shapely, PyProj |
| Raster Data Handling | Xarray, rioxarray |
| Image Processing | OpenCV, scikit-image, Pillow |
| Data Analysis | NumPy, Pandas, SciPy |
| Visualization | Folium, Streamlit-Folium, Plotly, Leafmap |
| Machine Learning Utilities | Scikit-learn |
| Experiment Tracking | MLflow |
| Testing | Pytest |

## Running Locally

If you would like to explore or run FloodSight locally, follow these steps.

### Prerequisites

- Python installed on your system.
- Git.
- Sufficient memory and storage for geospatial data and model dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/zobayerzo/floodsight.git
cd floodsight
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
streamlit run app.py
```

Once the application starts, Streamlit will provide a local URL in your terminal. Open that address in your browser to access FloodSight.

**Note:** Running the application locally may require the relevant model weights and geospatial datasets. Installing the Python dependencies alone may not provide all the files required for every analysis workflow.

## Important Considerations

FloodSight is designed as a research and analytical platform. Its outputs should be interpreted with appropriate care.

- **Water detection is not always equivalent to flood detection.** A predicted water region may include permanent water bodies or other water-related features.
- **Uncertainty estimates are not calibrated probabilities of error.** MC Dropout provides an uncertainty proxy that can help identify less consistent predictions.
- **Population exposure is an estimate.** It represents spatial overlap with population-related data, not a verified number of flood victims.
- **Geospatial datasets have temporal limitations.** For example, ESA WorldCover 2021 may not represent land-cover conditions at the time of an earlier flood event.
- **Model performance can vary.** Geographic differences, satellite imagery conditions, and unseen flood events may affect prediction quality.

FloodSight should therefore be considered a decision-support and exploratory analysis tool, not a replacement for official flood monitoring systems or emergency response authorities.

## Project Motivation

FloodSight was developed to explore how deep learning and geospatial technologies can work together to address a real-world environmental challenge.

The project brings together several areas of interest: computer vision, satellite imagery, semantic segmentation, uncertainty estimation, and geospatial data analysis.

Beyond generating a flood prediction, the project focuses on making those predictions easier to explore and understand through interactive maps and exposure analysis.

## Future Improvements

Potential directions for further development include:

- Improving model generalization across geographical regions and flood events.
- Exploring additional satellite data sources and segmentation approaches.
- Improving uncertainty estimation and its interpretation.
- Expanding population and infrastructure exposure analysis.
- Adding more comprehensive model evaluation and validation workflows.

## Author

**Md. Zobayer**

Computer Science and Engineering graduate interested in Machine Learning, Artificial Intelligence, Computer Vision, and Geospatial Data Science.

- GitHub: [@zobayerzo](https://github.com/zobayerzo)
- Project: [FloodSight](https://github.com/zobayerzo/floodsight)
- Live Demo: [FloodSight Web Application](https://floodsight-3.streamlit.app/)

---

*FloodSight — Exploring the intersection of satellite imagery, deep learning, and geospatial intelligence for flood analysis.*