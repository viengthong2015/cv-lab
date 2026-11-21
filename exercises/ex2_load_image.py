"""
ບົດຝຶກຫັດ 2: ທົດລອງ Load ຮູບພາບ
C# (EmguCV)                                  ->  Python (OpenCV + Streamlit)
-----------------------------------------------------------------------------
Button btnLoad + OpenFileDialog              ->  st.file_uploader
Image<Bgr, Byte> img = new Image<...>(file)  ->  cv2.imdecode(..., IMREAD_COLOR)  (BGR, uint8)
imageBox1.Image = img                        ->  st.image(img, channels="BGR")
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from code_viewer import show_code_button  # noqa: E402
from ui import page_header  # noqa: E402

st.set_page_config(page_title="Ex2 - Load Image", layout="wide")
page_header(2)
st.caption("ທຽບເທົ່າ C#: `btnLoad` (Button) + `imageBox1` (ImageBox)")
show_code_button(__file__)


# >>> CODE: ອ່ານ file ເປັນຮູບ BGR  (C#: new Image<Bgr, Byte>(opf.FileName))
def read_bgr(uploaded) -> np.ndarray | None:
    """ອ່ານ file ທີ່ upload ມາເປັນ BGR image (ຄື Image<Bgr, Byte> ໃນ EmguCV)."""
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
# <<< CODE


# >>> CODE: btnLoad_Click + imageBox1  (Load ແລະ ສະແດງຮູບ)
# --- btnLoad ---
uploaded = st.file_uploader("Load", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

# --- imageBox1 ---
if uploaded is None:
    st.info("ກົດ **Browse files** (ຫຼື ລາກຮູບມາວາງ) ເພື່ອ Load ຮູບພາບ.")
else:
    img = read_bgr(uploaded)
    if img is None:
        st.error("ບໍ່ສາມາດອ່ານ file ນີ້ເປັນຮູບພາບໄດ້.")
    else:
        h, w, c = img.shape
        st.image(img, channels="BGR", caption=uploaded.name)
        st.write(f"ຂະໜາດ: **{w} × {h}** pixels, **{c}** channels (BGR), dtype = `{img.dtype}`")
# <<< CODE
