"""
ບົດຝຶກຫັດ 3: Convert to Grayscale
C# (EmguCV)                                   ->  Python (OpenCV + Streamlit)
------------------------------------------------------------------------------
btnLoadImage + pictBoxOrigin                  ->  st.file_uploader + Plotly image
pictBoxOrigin_MouseMove (lblX, lblY, lblValue)->  hover ເທິງຮູບ (ສະແດງ X, Y, Value)
                                                  + ປ້ອນ X, Y ເພື່ອອ່ານຄ່າ pixel
My_Image[e.Y, e.X]                            ->  my_image[y, x]  (ລຳດັບ [B, G, R])
btnConvertToGray: My_Image.Convert<Gray,Byte>  ->  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
btnSaveImage + SaveFileDialog                 ->  st.download_button (jpg / png / bmp)
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from code_viewer import show_code_button  # noqa: E402
from ui import page_header  # noqa: E402

st.set_page_config(page_title="Ex3 - Convert to Gray", layout="wide")
page_header(3)
show_code_button(__file__)

MAX_DISPLAY_SIDE = 800  # ຮູບໃຫຍ່ຈະຖືກ sample ລົງເພື່ອໃຫ້ hover ໄວ (ພິກັດ X, Y ຍັງເປັນຂອງຮູບແທ້)


# >>> CODE: btnLoadImage_Click: ອ່ານ file ເປັນຮູບ BGR  (C#: My_Image = new Image<Bgr, Byte>(...))
def read_bgr(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
# <<< CODE


# >>> CODE: pictBoxOrigin_MouseMove -> hover ເທິງຮູບສະແດງ X, Y, Value
def image_figure(img: np.ndarray):
    """ສ້າງຮູບແບບ interactive: ເອົາ mouse ຊີ້ເທິງຮູບຈະເຫັນ X, Y ແລະ ຄ່າ pixel (ຄື MouseMove)."""
    h, w = img.shape[:2]
    step = max(1, math.ceil(max(h, w) / MAX_DISPLAY_SIDE))
    small = img[::step, ::step]
    xs, ys = np.arange(0, w, step), np.arange(0, h, step)

    if img.ndim == 2:  # Gray
        fig = px.imshow(small, x=xs, y=ys, color_continuous_scale="gray", zmin=0, zmax=255)
        fig.update_traces(hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
    else:  # Bgr -> ສະແດງເປັນ RGB
        fig = px.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), x=xs, y=ys)
        fig.update_traces(hovertemplate="X: %{x}<br>Y: %{y}<br>[R, G, B]: %{z}<extra></extra>")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420)
    return fig
# <<< CODE


# >>> CODE: Encode ຮູບເປັນ jpg / png / bmp  (ໃຊ້ຕອນ Save)
def encode(img: np.ndarray, ext: str) -> bytes:
    ok, buf = cv2.imencode(ext, img)
    return buf.tobytes() if ok else b""
# <<< CODE


# --- ຕົວປ່ຽນ (ຄື My_Image ແລະ gray_image ໃນ C#) ---
ss = st.session_state
ss.setdefault("ex3_file_id", None)
ss.setdefault("gray_image", None)

# --- btnLoadImage ---
uploaded = st.file_uploader("Load", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
if uploaded is None:
    st.info("ກົດ **Browse files** ເພື່ອ Load ຮູບພາບ.")
    st.stop()

if ss.ex3_file_id != uploaded.file_id:  # Load ຮູບໃໝ່ -> ລ້າງຮູບເທົາເກົ່າ
    ss.ex3_file_id = uploaded.file_id
    ss.gray_image = None

my_image = read_bgr(uploaded)  # Image<Bgr, Byte> My_Image
if my_image is None:
    st.error("ບໍ່ສາມາດອ່ານ file ນີ້ເປັນຮູບພາບໄດ້.")
    st.stop()
h, w = my_image.shape[:2]

# --- ປຸ່ມ: Convert to Grayscale / Save ---
b1, b2, b3, _ = st.columns([1.2, 0.8, 1, 2])
# >>> CODE: btnConvertToGray_Click  (C#: gray_image = My_Image.Convert<Gray, Byte>())
if b1.button("Convert to Grayscale", type="primary", width="stretch"):
    ss.gray_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)  # My_Image.Convert<Gray, Byte>()
# <<< CODE

# >>> CODE: btnSaveImage_Click  (C#: SaveFileDialog + gray_image.Save(...))
fmt = b2.selectbox("ຮູບແບບ file", ["jpg", "png", "bmp"], label_visibility="collapsed")
b3.download_button(
    "Save",
    data=encode(ss.gray_image, "." + fmt) if ss.gray_image is not None else b"",
    file_name=f"gray_image.{fmt}",
    mime=f"image/{'jpeg' if fmt == 'jpg' else fmt}",
    disabled=ss.gray_image is None,
    on_click=lambda: st.toast("Saving Complete"),
    width="stretch",
)
# <<< CODE

# --- pictBoxOrigin / pictBoxGray ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("pictBoxOrigin")
    st.plotly_chart(image_figure(my_image), width="stretch")
with c2:
    st.subheader("pictBoxGray")
    if ss.gray_image is None:
        st.info("ກົດປຸ່ມ **Convert to Grayscale**.")
    else:
        st.plotly_chart(image_figure(ss.gray_image), width="stretch")

# --- lblX, lblY, lblValue (ຄື pictBoxOrigin_MouseMove) ---
st.caption("💡 ເອົາ mouse ຊີ້ເທິງຮູບເພື່ອເບິ່ງ X, Y, Value — ຫຼື ປ້ອນພິກັດລຸ່ມນີ້:")
i1, i2, i3 = st.columns([1, 1, 3])
# >>> CODE: lblX, lblY, lblValue: ອ່ານຄ່າ pixel  (C#: My_Image[e.Y, e.X])
x = i1.number_input("X", min_value=0, max_value=w - 1, value=0, step=1)
y = i2.number_input("Y", min_value=0, max_value=h - 1, value=0, step=1)
b, g, r = my_image[y, x]  # My_Image[e.Y, e.X]  -> ລຳດັບແມ່ນ [B, G, R] ຄື EmguCV
value_text = f"Value: [{b},{g},{r}]  (B, G, R)"
if ss.gray_image is not None:
    value_text += f"   |   Gray: [{ss.gray_image[y, x]}]"
i3.markdown(f"**X:** {x} &nbsp;&nbsp; **Y:** {y}<br>**{value_text}**", unsafe_allow_html=True)
# <<< CODE
st.caption(f"ຂະໜາດຮູບ: {w} × {h} pixels")
