"""
Computer Vision Lab — ແປງບົດຝຶກຫັດ EmguCV (C#) ມາເປັນ Python + OpenCV + Streamlit

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Computer Vision Lab", page_icon="🖼️", layout="wide")

pages = [
    st.Page("exercises/ex2_load_image.py", title="ບົດຝຶກຫັດ 2: Load Image", icon="📂", default=True),
    st.Page("exercises/ex3_convert_gray.py", title="ບົດຝຶກຫັດ 3: Convert to Gray", icon="⚫"),
    st.Page("exercises/ex4_histogram.py", title="ບົດຝຶກຫັດ 4: Histogram", icon="📊"),
    st.Page("exercises/ex5_thresholding.py", title="ບົດຝຶກຫັດ 5: Thresholding", icon="🔲"),
    st.Page("exercises/ex6_equalization_filtering.py", title="ບົດຝຶກຫັດ 6: Equalization & Filtering", icon="🎚️"),
    st.Page("exercises/ex7_webcam.py", title="ບົດຝຶກຫັດ 7: WebCam", icon="📷"),
    st.Page("exercises/ex8_edge_detection.py", title="ບົດຝຶກຫັດ 8: Edge Detection", icon="✏️"),
]

st.navigation(pages).run()
