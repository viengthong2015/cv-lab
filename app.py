"""
Computer Vision Lab — ແປງບົດຝຶກຫັດ EmguCV (C#) ມາເປັນ Python + OpenCV + Streamlit

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Computer Vision Lab", page_icon=":material/camera:", layout="wide")

pages = [
    st.Page("exercises/home.py", title="ໜ້າຫຼັກ", icon=":material/home:", default=True),
    st.Page("exercises/ex2_load_image.py", title="ບົດຝຶກຫັດ 2: Load Image", icon=":material/image:"),
    st.Page("exercises/ex3_convert_gray.py", title="ບົດຝຶກຫັດ 3: Convert to Gray", icon=":material/filter_b_and_w:"),
    st.Page("exercises/ex4_histogram.py", title="ບົດຝຶກຫັດ 4: Histogram", icon=":material/bar_chart:"),
    st.Page("exercises/ex5_thresholding.py", title="ບົດຝຶກຫັດ 5: Thresholding", icon=":material/contrast:"),
    st.Page("exercises/ex6_equalization_filtering.py", title="ບົດຝຶກຫັດ 6: Equalization & Filtering",
            icon=":material/tune:"),
    st.Page("exercises/ex7_webcam.py", title="ບົດຝຶກຫັດ 7: WebCam", icon=":material/videocam:"),
    st.Page("exercises/ex8_morphology.py", title="ບົດຝຶກຫັດ 8: Morphology", icon=":material/category:"),
    st.Page("exercises/ex9_edge_detection.py", title="ບົດຝຶກຫັດ 9: Edge Detection", icon=":material/line_style:"),
    st.Page("exercises/ex10_template_matching.py", title="ບົດຝຶກຫັດ 10: Template Matching",
            icon=":material/center_focus_strong:"),
    st.Page("exercises/ex11_feature_matching.py", title="ບົດຝຶກຫັດ 11: ORB & SIFT Matching", icon=":material/hub:"),
]

st.navigation(pages).run()
