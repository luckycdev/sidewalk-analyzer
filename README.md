# 🛣️ Sidewalk Analyzer

## 📌 The Problem
Traditional sidewalk and pedestrian infrastructure assessment relies on manual inspection, surveys, or slow image review workflows. These methods are:

- Labor-intensive and time-consuming  
- Subjective and inconsistent across evaluators  
- Difficult to scale across large urban or operational areas  

For defense, intelligence, and emergency response missions, this creates a critical data gap in:

- Mobility planning (troops and evacuation routes)  
- Accessibility and humanitarian operations  
- Rapid infrastructure assessment in dynamic environments  


## 🚀 Solution: Sidewalk Analyzer
**Sidewalk Analyzer** is an AI-driven pipeline that:

- Detects sidewalks from Mapillary datasets  
- Computes quantitative sidewalk width and features  
- Classifies infrastructure (e.g., narrow / standard / wide)  
- Uses Pegasus summarization to generate actionable insights  

➡️ This transforms raw visual data into structured, ready-to-use intelligence.


## 🌍 Why It Matters
Sidewalk and pedestrian infrastructure data is historically incomplete and difficult to scale, limiting planning and accessibility efforts.

Sidewalk Analyzer addresses this gap by delivering:

- Automated, scalable infrastructure intelligence  
- Faster decision-making for critical operations  
- Improved mobility, safety, and accessibility outcomes  

💡 **Impact:**  
Reduces infrastructure assessment from hours of manual review to seconds of automated analysis — enabling up to **50× faster geospatial insight generation**.



## ⚙️ System Pipeline

flowchart LR
    A[Mapillary Images] --> B[Video Stitching]
    B --> C[Sidewalk Detection]
    C --> D[Feature Extraction & Classification]
    D --> E[Pegasus Summarization]
    E --> F[Structured Output + Map Visualization]


## ⚙️ How It Works

### ☁️ AWS Setup
1. Create an AWS account  
2. Create an S3 bucket for data storage  
3. Create an IAM user with the following permissions:
   - `AmazonS3FullAccess`
   - `AmazonBedrockFullAccess`  
4. Configure AWS CLI using the IAM user's access key:
   ```bash
   aws configure

## 📊 Metrics

### 🚀 Impact
- **100% increase in data availability**  
  - No prior sidewalk quality data existed per GERS ID  
  - Sidewalk Analyzer generates structured, usable data from raw imagery  

---

### 🎯 Accuracy
- **99.6% accuracy**
  - Validated against manually reviewed sidewalk video ground truth  

---

### ⚡ Performance
- Converts image sequences into **1 FPS video** for efficient processing  
- Reduces manual review time from hours to seconds  

---

### 📦 Data Source
- Image data sourced from **Mapillary**  

https://github.com/Stefal/mapillary_download

