"""
ບົດຝຶກຫັດ 4: ImageBox ແລະ HistogramBox
C# (EmguCV)                                   ->  Python (OpenCV + Streamlit)
------------------------------------------------------------------------------
btnLoadImage + imgInput                       ->  st.file_uploader + st.image
B = oriImage[0]; G = oriImage[1]; R = [2]     ->  B, G, R = cv2.split(img)
radioR / radioG / radioB (GroupBox Histogram) ->  st.radio
ComboBox "ເລືອກແມ່ສີ" (ວຽກບ້ານ 1)               ->  st.selectbox
imgOutput.Image = R                           ->  st.image(R)   (ຮູບເທົາຂອງ channel ນັ້ນ)
histogramBox1.GenerateHistograms(R, 256)      ->  cv2.calcHist(...) + Plotly bar chart
MessageBox.Show("Gray from Red Channel")      ->  st.toast(...)
Form1_Load: radioB.Checked = true             ->  ຄ່າເລີ່ມຕົ້ນ = Blue
ຕາຕະລາງເມັດສີ (pixel table)                    ->  img[y0:y1, x0:x1] + pandas DataFrame (ສີພື້ນ = ສີ pixel)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from code_viewer import show_code_button  # noqa: E402

st.set_page_config(page_title="Ex4 - Histogram", layout="wide")
st.title("ບົດຝຶກຫັດ 4: ທົດລອງນຳໃຊ້ ImageBox ແລະ HistogramBox")
show_code_button(__file__)

CHANNELS = ["Red", "Green", "Blue"]
COLORS = {"Red": "#e5484d", "Green": "#30a46c", "Blue": "#3e63dd"}


def read_bgr(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# >>> CODE: histogramBox1.GenerateHistograms(img, 256) -> cv2.calcHist + bar chart
def histogram_figure(channel_img: np.ndarray, name: str):
    """ຄື histogramBox1.GenerateHistograms(img, 256)."""
    hist = cv2.calcHist([channel_img], [0], None, [256], [0, 256]).ravel()
    fig = go.Figure(go.Bar(x=np.arange(256), y=hist, marker_color=COLORS[name], marker_line_width=0,
                           hovertemplate="Intensity: %{x}<br>Pixels: %{y:,}<extra></extra>"))
    fig.update_layout(
        title=f"Histogram — {name} channel",
        xaxis_title="Intensity (0–255)", yaxis_title="ຈຳນວນ pixel",
        xaxis_range=[-1, 256], bargap=0, height=330, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
# <<< CODE


# >>> CODE: ຕາຕະລາງເມັດສີ (pixel table) ຂະໜາດ rows × columns ຂອງຮູບ
STYLE_LIMIT = 40_000  # ໃສ່ສີພື້ນໄດ້ສະເພາະຕາຕະລາງ ≤ 40,000 ຊ່ອງ (ຕາຕະລາງໃຫຍ່ໃສ່ສີແລ້ວຊ້າຫຼາຍ)


def _frame(channel: np.ndarray) -> pd.DataFrame:
    """ແປງ matrix 2 ມິຕິ (rows × columns) ເປັນຕາຕະລາງ: ແຖວ = Row (Y), ຖັນ = Column (X)."""
    rows, cols = channel.shape
    return pd.DataFrame(channel, index=pd.RangeIndex(rows, name="Row (Y) \\ Col (X)"),
                        columns=pd.RangeIndex(cols))


@st.cache_data(show_spinner="ກຳລັງສ້າງຕາຕະລາງເມັດສີ...")
def color_pixel_table(img: np.ndarray) -> pd.DataFrame:
    """ຮູບສີ rows × columns: ຊ່ອງ [Y, X] = "B,G,R" ຂອງ pixel img[Y, X] (ລຳດັບຄື EmguCV)."""
    B, G, R = (_frame(img[:, :, i]).astype(str) for i in range(3))
    return B + "," + G + "," + R


@st.cache_data(show_spinner="ກຳລັງສ້າງຕາຕະລາງເມັດສີ...")
def gray_pixel_table(gray: np.ndarray) -> pd.DataFrame:
    """ຮູບເທົາ rows × columns: ຊ່ອງ [Y, X] = ຄ່າຄວາມເຂັ້ມ (0–255) ຂອງ pixel gray[Y, X]."""
    return _frame(gray)


def colorize(table: pd.DataFrame, img: np.ndarray):
    """ໃສ່ສີພື້ນຂອງແຕ່ລະຊ່ອງ = ສີຂອງ pixel ນັ້ນ (ຕົວອັກສອນດຳ/ຂາວ ຕາມຄວາມແຈ້ງ)."""
    bgr = img if img.ndim == 3 else np.dstack([img] * 3)
    B, G, R = (_frame(bgr[:, :, i]).astype(str) for i in range(3))
    lum = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
    fg = _frame(np.where(lum > 127, "black", "white"))
    css = "background-color: rgb(" + R + "," + G + "," + B + "); color: " + fg
    return table.style.apply(lambda _: css, axis=None)
# <<< CODE


def notify_channel(key: str):
    """ຄື MessageBox.Show("Gray from ... Channel") ເມື່ອປ່ຽນແມ່ສີ."""
    if st.session_state.get(key):
        st.toast(f"Gray from {st.session_state[key]} Channel")


main, side = st.columns([4, 1])

# ---------- ແຖບຄວບຄຸມດ້ານຂວາ ----------
with side:
    # btnLoadImage
    uploaded = st.file_uploader("Load Image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

    mode = st.segmented_control(
        "ວິທີເລືອກແມ່ສີ", ["RadioButton", "ComboBox"], default="RadioButton",
        help="RadioButton = ບົດຝຶກຫັດ 4,  ComboBox = ວຽກບ້ານ 1",
    ) or "RadioButton"

    # >>> CODE: ເລືອກແມ່ສີ: RadioButton (ບົດຝຶກຫັດ 4) ຫຼື ComboBox (ວຽກບ້ານ 1)
    if mode == "RadioButton":
        # GroupBox1 "Histogram" + radioR / radioG / radioB  (ຄ່າເລີ່ມຕົ້ນ: Blue)
        with st.container(border=True):
            selected = st.radio("Histogram", CHANNELS, index=2, key="radio_channel",
                                on_change=notify_channel, args=("radio_channel",))
    else:
        # ວຽກບ້ານ 1: ໃຊ້ ComboBox ແທນ RadioButton
        with st.container(border=True):
            selected = st.selectbox("ເລືອກແມ່ສີ", CHANNELS, index=None, placeholder="ເລືອກ...",
                                    key="combo_channel", on_change=notify_channel, args=("combo_channel",))
    # <<< CODE

# ---------- imgInput / imgOutput / histogramBox1 ----------
with main:
    if uploaded is None:
        st.info("ກົດ **Load Image** (ດ້ານຂວາ) ເພື່ອເລືອກຮູບພາບ.")
        st.stop()

    ori_image = read_bgr(uploaded)
    if ori_image is None:
        st.error("ບໍ່ສາມາດອ່ານ file ນີ້ເປັນຮູບພາບໄດ້.")
        st.stop()

    # >>> CODE: ແຍກ channel  (C#: B = oriImage[0]; G = oriImage[1]; R = oriImage[2])
    B, G, R = cv2.split(ori_image)  # B = oriImage[0], G = oriImage[1], R = oriImage[2]
    channel_map = {"Red": R, "Green": G, "Blue": B}
    # <<< CODE

    # ແຈ້ງເຕືອນຕອນ Load ຮູບໃໝ່ (ຄື MessageBox ໃນ btnLoadImage_Click)
    if st.session_state.get("ex4_file_id") != uploaded.file_id:
        st.session_state.ex4_file_id = uploaded.file_id
        if selected:
            st.toast(f"Gray from {selected} Channel")

    # >>> CODE: ສະແດງ imgInput, imgOutput (ຮູບເທົາຂອງແມ່ສີທີ່ເລືອກ) ແລະ histogramBox1
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("imgInput")
        st.image(ori_image, channels="BGR", width="stretch")
    with c2:
        st.subheader("imgOutput")
        if selected:
            st.image(channel_map[selected], width="stretch",
                     caption=f"Gray from {selected} Channel")
        else:
            st.info("ເລືອກແມ່ສີຈາກ ComboBox.")

    st.subheader("histogramBox1")
    if selected:
        st.plotly_chart(histogram_figure(channel_map[selected], selected), width="stretch")
    # <<< CODE

    # ---------- ຕາຕະລາງເມັດສີ (pixel table) ຂອງ 2 ຮູບ ----------
    st.divider()
    st.subheader("ຕາຕະລາງເມັດສີ (Pixel Table)")
    h, w = ori_image.shape[:2]

    # >>> CODE: ສະແດງ pixel table (rows × columns) ຂອງຮູບສີ ແລະ ຮູບເທົາ
    p1, p2 = st.columns([2, 1])
    gray_options = (["imgOutput"] if selected else []) + ["Convert to Gray"]
    gray_src = p1.radio("ຮູບເທົາ", gray_options, horizontal=True, captions=(
        [f"Gray from {selected} Channel"] if selected else []) + ["cv2.cvtColor(BGR2GRAY)"])
    gray_image = (channel_map[selected] if gray_src == "imgOutput"
                  else cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY))

    rows, cols = gray_image.shape          # rows = ຄວາມສູງ (Y), columns = ຄວາມກວ້າງ (X)
    n_cells = rows * cols
    use_color = p2.toggle("ໃສ່ສີພື້ນຕາມຄ່າ pixel", value=n_cells <= STYLE_LIMIT,
                          disabled=n_cells > STYLE_LIMIT,
                          help=f"ໃຊ້ໄດ້ເມື່ອຮູບມີ ≤ {STYLE_LIMIT:,} pixels (ຕາຕະລາງໃຫຍ່ໃສ່ສີແລ້ວຊ້າຫຼາຍ)")
    st.caption(f"ຕາຕະລາງ = **{rows} rows × {cols} columns = {n_cells:,} ຊ່ອງ** (1 ຊ່ອງ = 1 pixel). "
               f"ແຖວ = Row (Y: 0–{rows - 1}), ຖັນ = Column (X: 0–{cols - 1}), ຊ່ອງ [Y, X] = img[Y, X]. "
               "ເລື່ອນ (scroll) ຕາຕະລາງເພື່ອເບິ່ງທັງໝົດ.")

    # 1) ຮູບສີ (Color image): rows × columns × 3 channels -> ຄ່າແຕ່ລະຊ່ອງ = B,G,R
    st.markdown(f"**1. ຮູບສີ (Color Image) — {rows} × {cols} × 3, ຄ່າແຕ່ລະຊ່ອງ = `B,G,R`**")
    color_table = color_pixel_table(ori_image)
    st.dataframe(colorize(color_table, ori_image) if use_color else color_table, height=400)

    # 2) ຮູບເທົາ (Gray image): rows × columns -> ຄ່າແຕ່ລະຊ່ອງ = Intensity 0–255
    st.markdown(f"**2. ຮູບເທົາ (Gray Image) — {rows} × {cols}, ຄ່າແຕ່ລະຊ່ອງ = Intensity (0–255)**")
    gray_table = gray_pixel_table(gray_image)
    st.dataframe(colorize(gray_table, gray_image) if use_color else gray_table, height=400)
    # <<< CODE
