"""
ບົດຝຶກຫັດ 6: ການປັບຄ່າຟິກເຊວຂອງຮູບພາບ ແລະ ການຕອງຮູບພາບ (Histogram Equalization and Image Filtering)
C# (EmguCV)                                          ->  Python (OpenCV + Streamlit)
--------------------------------------------------------------------------------------
btnLoadImage + OpenFileDialog                        ->  st.file_uploader
new Image<Gray, byte>(file) / imgBoxInput            ->  cv2.imdecode + cv2.cvtColor(BGR2GRAY) / st.image
htgBeforeE.GenerateHistograms(img, 255)              ->  cv2.calcHist(...) + Plotly
CvInvoke.EqualizeHist(imgBeforeE, imgAfterE)         ->  cv2.equalizeHist(gray)
CvInvoke.GaussianBlur(img, out, Size(7,7), 0, 0, ..) ->  cv2.GaussianBlur(img, (k, k), 0)
CvInvoke.MedianBlur(img, out, 7)                     ->  cv2.medianBlur(img, k)
btnSaveImage + SaveFileDialog (jpg / bmp)            ->  st.download_button
btnExit: this.Close()                                ->  (web app: ປິດ tab / Ctrl+C ໃນ terminal)

ວຽກບ້ານ:
  ImageBox1 = ຮູບສີທີ່ Load ເຂົ້າມາ,  ImageBox2 = ຮູບສີເທົາຂອງຮູບທີ 1,
  ImageBox3 = 1) ຜົນ Histogram Equalization ຂອງຮູບສີເທົາ
              2) ຜົນກຳຈັດ Noise (Gaussian / Median) ຂອງ "ຮູບສີ" ຫຼື "ຮູບສີເທົາ" (ເລືອກດ້ວຍ Radio Button)
              3) ksize ປ່ຽນໄດ້ຈາກ TextBox (ເລກຄີກ ≥ 3)
              4) ປຸ່ມ Save ແລະ Exit
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from code_viewer import show_code_button  # noqa: E402

st.set_page_config(page_title="Ex6 - Equalization & Filtering", layout="wide")
st.title("ບົດຝຶກຫັດ 6: ການປັບຄ່າຟິກເຊວຂອງຮູບພາບ ແລະ ການຕອງຮູບພາບ")
st.caption("Histogram Equalization and Image Filtering — ລວມວຽກບ້ານ (ImageBox 3 ອັນ, Radio Button, ksize, Save, Exit)")
show_code_button(__file__)

SOURCES = ["ຮູບພາບສີ", "ຮູບພາບສີເທົາ", "ຮູບເທົາຫຼັງ Equalization (ຕາມ code C#)"]
OP_NAMES = {"equalize": "Histogram Equalization", "gaussian": "Gaussian Filter", "median": "Median Filter"}
CHANNEL_COLORS = [("Blue", "#3e63dd"), ("Green", "#30a46c"), ("Red", "#e5484d")]  # ລຳດັບ B, G, R


def read_bgr(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# >>> CODE: htgBeforeE / htgAfterE.GenerateHistograms(img, 255) -> cv2.calcHist + Plotly
def histogram_figure(img: np.ndarray, title: str):
    """ຮູບເທົາ -> histogram 1 ເສັ້ນ;  ຮູບສີ -> histogram 3 ເສັ້ນ (B, G, R)."""
    fig = go.Figure()
    if img.ndim == 2:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
        fig.add_bar(x=np.arange(256), y=hist, marker_color="#6f6f6f", marker_line_width=0, name="Gray",
                    hovertemplate="Intensity: %{x}<br>Pixels: %{y:,}<extra></extra>")
    else:
        for i, (name, color) in enumerate(CHANNEL_COLORS):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256]).ravel()
            fig.add_scatter(x=np.arange(256), y=hist, mode="lines", line_color=color, name=name,
                            hovertemplate=name + " %{x}: %{y:,} pixels<extra></extra>")
    fig.update_layout(title=title, xaxis_title="Intensity (0–255)", yaxis_title="ຈຳນວນ pixel",
                      xaxis_range=[-1, 256], bargap=0, height=300, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return fig
# <<< CODE


def parse_ksize(text: str, default: int = 7) -> int:
    """ອ່ານ ksize ຈາກ TextBox: ຕ້ອງເປັນເລກຄີກ ≥ 3 (ວຽກບ້ານ ຂໍ້ 3)."""
    try:
        k = int(text)
    except ValueError:
        st.warning(f"ksize ຕ້ອງເປັນຕົວເລກ — ໃຊ້ຄ່າ {default} ແທນ.")
        return default
    if k < 3:
        st.warning(f"ksize ຕ້ອງ ≥ 3 — ໃຊ້ຄ່າ 3 ແທນ.")
        return 3
    if k > 99:
        st.warning("ksize ໃຫຍ່ເກີນໄປ — ໃຊ້ຄ່າ 99 ແທນ.")
        return 99
    if k % 2 == 0:
        st.warning(f"ksize ຕ້ອງເປັນເລກຄີກ — ປ່ຽນເປັນ {k + 1}.")
        return k + 1
    return k


# >>> CODE: btnEqualization / btnGuassian / btnMedian: ສ້າງຮູບຜົນລັບ (ImageBox3)
def process(op: str, color_img, gray_img, source: str, k: int) -> np.ndarray:
    """ຄືກັບ btnEqualization_Click, btnGuassian_Click, btnMedian_Click ໃນ C#."""
    if op == "equalize":
        return cv2.equalizeHist(gray_img)                     # CvInvoke.EqualizeHist(imgBeforeE, imgAfterE)

    # Radio Button: ເລືອກວ່າຈະກຳຈັດ Noise ຈາກຮູບໃດ
    if source == SOURCES[0]:
        src = color_img                                       # ຮູບພາບສີ
    elif source == SOURCES[1]:
        src = gray_img                                        # ຮູບພາບສີເທົາ
    else:
        src = cv2.equalizeHist(gray_img)                      # imgAfterE (ຄື code C# ຕົ້ນສະບັບ)

    if op == "gaussian":
        return cv2.GaussianBlur(src, (k, k), 0, borderType=cv2.BORDER_DEFAULT)  # GaussianBlur(.., Size(k,k), 0, 0, ..)
    return cv2.medianBlur(src, k)                             # CvInvoke.MedianBlur(imgAfterE, imgFilter, k)
# <<< CODE


ss = st.session_state
ss.setdefault("ex6_file_id", None)
ss.setdefault("ex6_op", None)  # ປຸ່ມທີ່ກົດຫຼ້າສຸດ: equalize / gaussian / median

main, side = st.columns([4, 1])

# ---------- ແຜງຄວບຄຸມດ້ານຂວາ (ປຸ່ມຄືໃນ Form C#) ----------
with side:
    uploaded = st.file_uploader("Load Image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
    if uploaded is not None and ss.ex6_file_id != uploaded.file_id:  # Load ຮູບໃໝ່ -> ລ້າງຜົນເກົ່າ
        ss.ex6_file_id = uploaded.file_id
        ss.ex6_op = None
    no_image = uploaded is None

    # >>> CODE: ປຸ່ມ ແລະ Radio Button / TextBox ksize (ວຽກບ້ານ ຂໍ້ 2, 3)
    if st.button("Histogram Equalization", width="stretch", disabled=no_image):
        ss.ex6_op = "equalize"

    with st.container(border=True):
        source = st.radio("ກຳຈັດ Noise ຈາກ", SOURCES, index=1)
        ksize_text = st.text_input("ksize (ເລກຄີກ ≥ 3)", value="7")
        if st.button("Guassian Filter", width="stretch", disabled=no_image):
            ss.ex6_op = "gaussian"
        if st.button("Median Filter", width="stretch", disabled=no_image):
            ss.ex6_op = "median"
    k = parse_ksize(ksize_text)
    # <<< CODE

if no_image:
    with main:
        st.info("ກົດ **Load Image** (ດ້ານຂວາ) ເພື່ອເລືອກຮູບພາບ.")
    st.stop()

# >>> CODE: btnLoadImage_Click: ອ່ານຮູບສີ (ImageBox1) ແລະ ປ່ຽນເປັນຮູບສີເທົາ (ImageBox2)
color_img = read_bgr(uploaded)
if color_img is None:
    with main:
        st.error("ບໍ່ສາມາດອ່ານ file ນີ້ເປັນຮູບພາບໄດ້.")
    st.stop()
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  # imgBeforeE = new Image<Gray, byte>(file)
# <<< CODE

output = process(ss.ex6_op, color_img, gray_img, source, k) if ss.ex6_op else None

# ---------- ImageBox1 / ImageBox2 / ImageBox3 + htgBeforeE / htgAfterE ----------
with main:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**ImageBox1 — ຮູບສີ (Input)**")
        st.image(color_img, channels="BGR", width="stretch")
    with c2:
        st.markdown("**ImageBox2 — ຮູບສີເທົາ**")
        st.image(gray_img, width="stretch")
    with c3:
        st.markdown("**ImageBox3 — ຜົນລັບ (Output)**")
        if output is None:
            st.info("ກົດ **Histogram Equalization**, **Guassian Filter** ຫຼື **Median Filter**.")
        else:
            caption = OP_NAMES[ss.ex6_op]
            if ss.ex6_op != "equalize":
                caption += f" | ksize = {k} | ຈາກ: {source}"
            st.image(output, channels="BGR" if output.ndim == 3 else "RGB", width="stretch", caption=caption)

    h1, h2 = st.columns(2)
    h1.plotly_chart(histogram_figure(gray_img, "htgBeforeE — ກ່ອນ (ຮູບສີເທົາ)"), width="stretch")
    if output is not None:
        h2.plotly_chart(histogram_figure(output, f"htgAfterE — ຫຼັງ ({OP_NAMES[ss.ex6_op]})"), width="stretch")

# ---------- btnSaveImage / btnExit (ວຽກບ້ານ ຂໍ້ 4) ----------
with side:
    # >>> CODE: btnSaveImage_Click: ບັນທຶກຮູບ ImageBox3 (jpg / bmp)
    s1, s2 = st.columns([1, 1.4])
    fmt = s1.selectbox("ຮູບແບບ file", ["jpg", "bmp"], label_visibility="collapsed")
    data = cv2.imencode("." + fmt, output)[1].tobytes() if output is not None else b""
    s2.download_button("Save Image", data=data, file_name=f"equalization_filtering.{fmt}",
                       mime="image/jpeg" if fmt == "jpg" else "image/bmp",
                       disabled=output is None, width="stretch",
                       on_click=lambda: st.toast("Saving Complete"))
    # <<< CODE
    st.caption("**Exit**: ໃນ web app ບໍ່ມີ this.Close() — ປິດ tab browser ຫຼື ກົດ Ctrl+C ໃນ terminal.")
