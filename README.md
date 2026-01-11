
---

# 🔬 Neural-Shape Analytics Hub (VisionDraft)

**Developer:** Naveen N (22MIA1049)

**Course:** Computer Vision

**Live Demo:** [https://naveen-visiondraft.streamlit.app/](https://naveen-visiondraft.streamlit.app/)

---

## 🚀 Project Overview

**Neural-Shape Analytics Hub** is a specialized Computer Vision application built to automate the identification and geometric analysis of various shapes. By leveraging **OpenCV** for image processing and **Streamlit** for the dashboard interface, this tool provides real-time feature extraction for industrial and academic research purposes.

## 🧠 Technical Features

* **Geometric Classification:** High-precision detection of **Triangles, Squares, Rectangles, Circles,** and complex **Polygons**.
* **Feature Extraction:** Generates data for every detected object, including:
* **Side Count:** Automated vertex calculation.
* **Area ():** Surface area calculation.
* **Perimeter ():** Boundary length.
* **Circularity:** A metric  indicating how close a shape is to a perfect circle.


* **Interactive Visual Insights:** A "Quad-Plot" dashboard using **Plotly** to visualize data distribution and geometric correlations.
* **Advanced Image Controls:** Real-time adjustments for Binary Sensitivity (Thresholding), Contour Colors, and Line Thickness.
* **Deliverable Export Hub:** One-click downloads for:
* The processed analysis view.
* The binary computer vision mask.
* Full technical feature reports in CSV format.



## 🛠️ Installation & Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

```


2. **Install the required libraries:**
```bash
pip install -r requirements.txt

```


3. **Run the application:**
```bash
streamlit run app.py

```



## 📂 Project Structure

* `app.py`: The main application logic and Streamlit UI.
* `requirements.txt`: Python library dependencies.
* `packages.txt`: System-level dependencies for OpenCV cloud deployment.
* `README.md`: Project documentation.

## 🎓 Academic Context

This project was developed as part of a Computer Vision lab experiment to demonstrate the practical application of **Contour Approximation** and **Moment-based Feature Extraction**.

---

