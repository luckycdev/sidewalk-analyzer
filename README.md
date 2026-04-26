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


what needs to be done:

How it works:
Set up
Any dataset documentation, preprocessing steps, and reproducibility notes
Explain how sidewalk analyzer works (technical documentation more in depth)
Where the data comes from
What it contains (images, video, etc.)
Any preprocessing steps
Which API/model you used (Pegasus)
What each model does
Why you chose it
What it does in your pipeline
How do you know something is a sidewalk?
Example output


aws bucket s3full bedrockfull
aws cli login
change aws_test.py vars

https://github.com/Stefal/mapillary_download

csv_creator.py

mp4_creator.py

aws_test.py
