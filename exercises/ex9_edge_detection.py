"""
ບົດຝຶກຫັດ 9: ການກວດຈັບຂອບ (Edge Detection)
C# (EmguCV)                                          ->  Python (OpenCV + Streamlit)
--------------------------------------------------------------------------------------
MenuStrip: File (Open, Save, Exit), Edge (Canny, Sobel, Laplacian)
                                                     ->  st.popover("File"), st.popover("Edge")
Image<Bgr, byte> imgInput = new Image<...>(file)     ->  cv2.imdecode(..., IMREAD_COLOR)
imgCanny = imgInput.Canny(50, 20)                    ->  cv2.Canny(img, 50, 20)                (uint8)
imgGray  = imgInput.Convert<Gray, byte>()            ->  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgSobel = imgGray.Sobel(1, 1, 3)                    ->  cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)  (float)
imgLaplacian = imgGray.Laplace(3)                    ->  cv2.Laplacian(gray, cv2.CV_32F, ksize=3)    (float)
imageBox1.Image = ...                                ->  st.image(...)  (float -> ແປງເປັນ 0–255 ກ່ອນສະແດງ)
saveToolStripMenuItem: SaveFileDialog (jpg/png/bmp) ->  st.download_button + st.toast("Saving Complete")
exitToolStripMenuItem: this.Close()                  ->  (web app: ປິດ tab / Ctrl+C ໃນ terminal)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from code_viewer import show_code_button  # noqa: E402
from ui import page_header  # noqa: E402

st.set_page_config(page_title="Ex9 - Edge Detection", layout="wide")
page_header(9)
show_code_button(__file__)

OP_NAMES = {"canny": "Canny", "sobel": "Sobel", "laplacian": "Laplacian"}
DISPLAY_MODES = ["Normalize (min–max)", "ຄ່າສົມບູນ |ຄ່າ| (convertScaleAbs)"]


def read_bgr(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# >>> CODE: Edge: Canny / Sobel / Laplacian  (ຄື cannyToolStripMenuItem_Click, sobel..., laplacian...)
def detect_edges(img_input: np.ndarray, op: str, p: dict) -> np.ndarray:
    """ສົ່ງຄືນ imgCanny (uint8) ຫຼື imgSobel / imgLaplacian (float32 ຄື Image<Gray, float>)."""
    if op == "canny":
        return cv2.Canny(img_input, p["canny_t1"], p["canny_t2"])               # imgInput.Canny(50, 20)

    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)                      # Convert<Gray, byte>()
    if op == "sobel":
        return cv2.Sobel(img_gray, cv2.CV_32F, p["dx"], p["dy"], ksize=p["sobel_k"])  # Sobel(1, 1, 3)
    return cv2.Laplacian(img_gray, cv2.CV_32F, ksize=p["lap_k"])                # Laplace(3)
# <<< CODE


# >>> CODE: ແປງຮູບ float (Image<Gray, float>) ເປັນ 0–255 ເພື່ອສະແດງ ແລະ Save
def to_display(edge: np.ndarray, mode: str) -> np.ndarray:
    """Sobel / Laplacian ມີຄ່າລົບ ແລະ ເກີນ 255 -> ຕ້ອງແປງກ່ອນສະແດງ.
    - Normalize: ຄ່າຕ່ຳສຸດ -> 0, ສູງສຸດ -> 255 (ພື້ນຫຼັງ = ສີເທົາ ຄືຮູບໃນ ImageBox ຂອງ C#)
    - ຄ່າສົມບູນ: |ຄ່າ| ແລ້ວຕັດທີ່ 255 (ພື້ນຫຼັງດຳ, ຂອບຂາວ)."""
    if edge.dtype == np.uint8:          # Canny ເປັນ 0 / 255 ຢູ່ແລ້ວ
        return edge
    if mode == DISPLAY_MODES[0]:
        return cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.convertScaleAbs(edge)
# <<< CODE


ss = st.session_state
ss.setdefault("ex9_file_id", None)
ss.setdefault("ex9_op", None)  # Edge ທີ່ເລືອກ: canny / sobel / laplacian (None = ສະແດງຮູບຕົ້ນສະບັບ)

# ---------- MenuStrip: File | Edge ----------
m1, m2, m3, _ = st.columns([1, 1, 2.2, 4])

with m1.popover("File", width="stretch"):
    # >>> CODE: File > Open  (openToolStripMenuItem_Click)
    uploaded = st.file_uploader("Open", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
    # <<< CODE
    save_slot = st.container()  # File > Save (ໃສ່ປຸ່ມພາຍຫຼັງ ເມື່ອມີຮູບຜົນລັບແລ້ວ)
    st.caption("**Exit**: ໃນ web app ບໍ່ມີ this.Close() — ປິດ tab browser ຫຼື ກົດ Ctrl+C ໃນ terminal.")

if uploaded is not None and ss.ex9_file_id != uploaded.file_id:  # Open ຮູບໃໝ່ -> ສະແດງຮູບຕົ້ນສະບັບ
    ss.ex9_file_id = uploaded.file_id
    ss.ex9_op = None

with m2.popover("Edge", width="stretch", disabled=uploaded is None):
    # >>> CODE: Edge > Canny / Sobel / Laplacian  (if (imgInput == null) return;)
    for key, name in OP_NAMES.items():
        if st.button(name, key=f"menu_{key}", width="stretch"):
            ss.ex9_op = key
    if st.button("ຮູບຕົ້ນສະບັບ (Input)", key="menu_input", width="stretch"):
        ss.ex9_op = None
    # <<< CODE

with m3.popover("⚙️ ປັບຄ່າ parameter", width="stretch"):
    st.caption("ຄ່າເລີ່ມຕົ້ນ = ຄ່າໃນ code C#")
    params = dict(
        canny_t1=st.slider("Canny: threshold1", 0, 500, 50),
        canny_t2=st.slider("Canny: threshold2", 0, 500, 20),
        dx=st.radio("Sobel: xorder (dx)", [0, 1, 2], index=1, horizontal=True),
        dy=st.radio("Sobel: yorder (dy)", [0, 1, 2], index=1, horizontal=True),
        sobel_k=st.radio("Sobel: ksize", [1, 3, 5, 7], index=1, horizontal=True),
        lap_k=st.radio("Laplacian: ksize", [1, 3, 5, 7], index=1, horizontal=True),
    )
    display_mode = st.radio("ສະແດງຮູບ float (Sobel / Laplacian)", DISPLAY_MODES, index=0)
    if params["dx"] == 0 and params["dy"] == 0:
        st.warning("Sobel: dx ແລະ dy ເປັນ 0 ພ້ອມກັນບໍ່ໄດ້ — ໃຊ້ dx = 1.")
        params["dx"] = 1

# ---------- imageBox1 ----------
if uploaded is None:
    st.info("ເປີດເມນູ **File → Open** ເພື່ອເລືອກຮູບພາບ.")
    st.stop()

img_input = read_bgr(uploaded)  # Image<Bgr, byte> imgInput
if img_input is None:
    st.error("ບໍ່ສາມາດອ່ານ file ນີ້ເປັນຮູບພາບໄດ້.")
    st.stop()

compare = st.toggle("ສະແດງຮູບຕົ້ນສະບັບທຽບຂ້າງກັນ", value=True)

if ss.ex9_op is None:
    shown, caption = img_input, f"imgInput — {uploaded.name}"
else:
    edge = detect_edges(img_input, ss.ex9_op, params)
    shown = to_display(edge, display_mode)
    caption = f"{OP_NAMES[ss.ex9_op]} — dtype {edge.dtype}"
    if edge.dtype != np.uint8:
        caption += f", ຄ່າຈິງ {edge.min():.0f} ຫາ {edge.max():.0f} → ສະແດງແບບ {display_mode}"

if compare and ss.ex9_op is not None:
    c1, c2 = st.columns(2)
    c1.markdown("**imgInput**")
    c1.image(img_input, channels="BGR", width="stretch")
    c2.markdown(f"**imageBox1 — {OP_NAMES[ss.ex9_op]}**")
    c2.image(shown, width="stretch", caption=caption)
else:
    st.markdown("**imageBox1**")
    st.image(shown, channels="BGR" if shown.ndim == 3 else "RGB", width="stretch", caption=caption)

# ---------- File > Save ----------
with save_slot:
    # >>> CODE: File > Save  (saveToolStripMenuItem_Click: jpg / png / bmp)
    fmt = st.selectbox("ຮູບແບບ file", ["jpg", "png", "bmp"])
    name = OP_NAMES.get(ss.ex9_op, "input").lower()
    st.download_button("Save", data=cv2.imencode("." + fmt, shown)[1].tobytes(),
                       file_name=f"{name}.{fmt}", mime=f"image/{'jpeg' if fmt == 'jpg' else fmt}",
                       width="stretch", on_click=lambda: st.toast("Saving Complete"))
    # <<< CODE
