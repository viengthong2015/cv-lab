"""
ບົດຝຶກຫັດ 8: ປະມວນຜົນພາບດ້ວຍ Morphology (Morphological Processing)
C# (EmguCV)                                               ->  Python (OpenCV + Streamlit)
-------------------------------------------------------------------------------------------
MenuStrip: File (Open, Save, Exit), Morphology (Dilation ... Black Hat, Morphological Binary ▸)
                                                          ->  st.popover("File"), st.popover("Morphology")
inputImage.Dilate(3)                                      ->  cv2.dilate(img, kernel 3×3, iterations=3)
inputImage.Erode(5)                                       ->  cv2.erode(img, kernel 3×3, iterations=5)
CvInvoke.GetStructuringElement(Rectangle, Size(3,3), Point(-1,-1))
                                                          ->  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
inputImage.MorphologyEx(MorphOp.Open, kernel, Point(-1,-1), 1, BorderType.Default, MCvScalar(1.0))
                                                          ->  cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, ...)
  MorphOp: Open / Close / Gradient / Tophat / Blackhat    ->  cv2.MORPH_OPEN / CLOSE / GRADIENT / TOPHAT / BLACKHAT
Morphological Binary:
  inputImage.Convert<Gray, byte>().ThresholdBinaryInv(Gray(150), Gray(255)).Dilate(1)
                                                          ->  cv2.threshold(gray, 150, 255, THRESH_BINARY_INV)
                                                              + cv2.dilate / erode / morphologyEx
exitToolStripMenuItem: this.Close()                       ->  (web app: ປິດ tab / Ctrl+C ໃນ terminal)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from code_viewer import show_code_button  # noqa: E402
from ui import page_header  # noqa: E402

st.set_page_config(page_title="Ex8 - Morphology", layout="wide")
page_header(8)
show_code_button(__file__)

# ລາຍການໃນເມນູ Morphology (key -> ຊື່ໃນເມນູ). "bin_" = ເມນູຍ່ອຍ Morphological Binary
COLOR_OPS = {"dilate": "Dilation", "erode": "Erosion", "open": "Opening", "close": "Closing",
             "gradient": "Gradient", "tophat": "Top Hat", "blackhat": "Black Hat"}
BINARY_OPS = {"bin_dilate": "Dilation", "bin_erode": "Erosion", "bin_open": "Opening", "bin_close": "Closing"}
DEFAULT_ITER = {"dilate": 3, "erode": 5}  # Dilate(3), Erode(5) ໃນ C#; ອັນອື່ນ = 1
SHAPES = {"Rectangle": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}
EXPLAIN = {
    "dilate": "Dilation: ແຕ່ລະ pixel = ຄ່າສູງສຸດໃນ kernel → ພື້ນທີ່ສະຫວ່າງຂະຫຍາຍອອກ, ຮູມືດນ້ອຍໆຫາຍໄປ.",
    "erode": "Erosion: ແຕ່ລະ pixel = ຄ່າຕ່ຳສຸດໃນ kernel → ພື້ນທີ່ສະຫວ່າງຫົດລົງ, ຈຸດສະຫວ່າງນ້ອຍໆຫາຍໄປ.",
    "open": "Opening = Erosion ແລ້ວ Dilation → ລົບຈຸດ noise ສະຫວ່າງນ້ອຍໆ ໂດຍຮັກສາຮູບຮ່າງວັດຖຸໃຫຍ່ໄວ້.",
    "close": "Closing = Dilation ແລ້ວ Erosion → ຖົມຮູ ແລະ ຮອຍແຕກມືດນ້ອຍໆ ໃນວັດຖຸ.",
    "gradient": "Gradient = Dilation − Erosion → ໄດ້ເສັ້ນຂອບຂອງວັດຖຸ.",
    "tophat": "Top Hat = ຮູບຕົ້ນສະບັບ − Opening → ເຫຼືອແຕ່ລາຍລະອຽດສະຫວ່າງທີ່ນ້ອຍກວ່າ kernel.",
    "blackhat": "Black Hat = Closing − ຮູບຕົ້ນສະບັບ → ເຫຼືອແຕ່ລາຍລະອຽດມືດທີ່ນ້ອຍກວ່າ kernel.",
}


def read_bgr(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# >>> CODE: kernel (Structuring Element)  (C#: CvInvoke.GetStructuringElement(Rectangle, Size(3,3), Point(-1,-1)))
def make_kernel(shape: str, size: int) -> np.ndarray:
    return cv2.getStructuringElement(SHAPES[shape], (size, size), (-1, -1))
# <<< CODE


# >>> CODE: Morphology: Dilation / Erosion / MorphologyEx (Open, Close, Gradient, Tophat, Blackhat)
MORPH_EX = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE, "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT, "blackhat": cv2.MORPH_BLACKHAT}


def apply_morphology(src: np.ndarray, op: str, kernel: np.ndarray, iterations: int) -> np.ndarray:
    """op ບໍ່ມີ "bin_" (ເຊັ່ນ "dilate", "open"). ໃຊ້ໄດ້ທັງຮູບສີ (BGR) ແລະ ຮູບຂາວດຳ."""
    if op == "dilate":
        return cv2.dilate(src, kernel, iterations=iterations)                 # inputImage.Dilate(3)
    if op == "erode":
        return cv2.erode(src, kernel, iterations=iterations)                  # inputImage.Erode(5)
    return cv2.morphologyEx(src, MORPH_EX[op], kernel, anchor=(-1, -1),       # MorphologyEx(MorphOp.X, kernel,
                            iterations=iterations, borderType=cv2.BORDER_DEFAULT,  # Point(-1,-1), 1,
                            borderValue=1.0)                                  # BorderType.Default, MCvScalar(1.0))
# <<< CODE


# >>> CODE: Morphological Binary: ຮູບເທົາ -> ThresholdBinaryInv(150, 255) -> Morphology
def to_binary(img_bgr: np.ndarray, thresh: int) -> np.ndarray:
    """inputImage.Convert<Gray, byte>().ThresholdBinaryInv(new Gray(150), new Gray(255)).
    Inv: ວັດຖຸສີເຂັ້ມ -> ສີຂາວ (255), ພື້ນຫຼັງແຈ້ງ -> ດຳ (0) — Morphology ຈຶ່ງເຮັດວຽກກັບວັດຖຸ."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)[1]
# <<< CODE


ss = st.session_state
ss.setdefault("ex8_file_id", None)
ss.setdefault("ex8_op", None)  # ເມນູທີ່ເລືອກ (None = ສະແດງຮູບຕົ້ນສະບັບ)

# ---------- MenuStrip: File | Morphology | parameter ----------
m1, m2, m3, _ = st.columns([1, 1.4, 2.2, 3.4])

with m1.popover("File", width="stretch"):
    # >>> CODE: File > Open  (openToolStripMenuItem_Click)
    uploaded = st.file_uploader("Open", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
    # <<< CODE
    save_slot = st.container()  # File > Save (ໃສ່ປຸ່ມພາຍຫຼັງ ເມື່ອມີຮູບຜົນລັບແລ້ວ)
    st.caption("**Exit**: ໃນ web app ບໍ່ມີ this.Close() — ປິດ tab browser ຫຼື ກົດ Ctrl+C ໃນ terminal.")

if uploaded is not None and ss.ex8_file_id != uploaded.file_id:  # Open ຮູບໃໝ່ -> ສະແດງຮູບຕົ້ນສະບັບ
    ss.ex8_file_id = uploaded.file_id
    ss.ex8_op = None

with m2.popover("Morphology", width="stretch", disabled=uploaded is None):
    # >>> CODE: ເມນູ Morphology  (if (inputImage != null) { ... })
    for key, name in COLOR_OPS.items():
        if st.button(name, key=f"menu_{key}", width="stretch"):
            ss.ex8_op = key
    st.markdown("**Morphological Binary**")
    for key, name in BINARY_OPS.items():
        if st.button(name, key=f"menu_{key}", width="stretch"):
            ss.ex8_op = key
    st.divider()
    if st.button("ຮູບຕົ້ນສະບັບ (Input)", key="menu_input", width="stretch"):
        ss.ex8_op = None
    # <<< CODE

op = ss.ex8_op
base_op = op.removeprefix("bin_") if op else None
is_binary = bool(op) and op.startswith("bin_")

with m3.popover("⚙️ ປັບຄ່າ parameter", width="stretch"):
    st.caption("ຄ່າເລີ່ມຕົ້ນ = ຄ່າໃນ code C#")
    shape = st.radio("ຮູບຮ່າງ kernel", list(SHAPES), horizontal=True)
    ksize = st.select_slider("ຂະໜາດ kernel", options=list(range(3, 22, 2)), value=3,
                             format_func=lambda k: f"{k}×{k}")
    # ຈຳນວນຮອບ: ແຍກຄ່າຕາມເມນູ (Dilation = 3, Erosion = 5, ອັນອື່ນ = 1 ຄື C#)
    iterations = st.number_input("ຈຳນວນຮອບ (iterations)", 1, 20, DEFAULT_ITER.get(op, 1),
                                 key=f"iter_{op}", disabled=op is None,
                                 help="ໃຊ້ກັບເມນູທີ່ເລືອກຢູ່ເທົ່ານັ້ນ")
    thresh = st.slider("Threshold ຂອງ Morphological Binary", 0, 255, 150, disabled=not is_binary)

# ---------- imgBoxInput / imgBoxOutput ----------
if uploaded is None:
    st.info("ເປີດເມນູ **File → Open** ເພື່ອເລືອກຮູບພາບ.")
    st.stop()

input_image = read_bgr(uploaded)  # Image<Bgr, byte> inputImage
if input_image is None:
    st.error("ບໍ່ສາມາດອ່ານ file ນີ້ເປັນຮູບພາບໄດ້.")
    st.stop()

if op is None:
    output, binary_image = None, None
else:
    kernel = make_kernel(shape, ksize)
    binary_image = to_binary(input_image, thresh) if is_binary else None
    src = binary_image if is_binary else input_image
    output = apply_morphology(src, base_op, kernel, int(iterations))

if output is None:
    st.markdown("**imgBoxInput**")
    st.image(input_image, channels="BGR", width="stretch", caption=uploaded.name)
    st.caption("ເລືອກຄຳສັ່ງຈາກເມນູ **Morphology**.")
else:
    title = ("Morphological Binary ▸ " if is_binary else "") + (BINARY_OPS | COLOR_OPS)[op]
    cols = st.columns(3 if is_binary else 2)
    cols[0].markdown("**imgBoxInput**")
    cols[0].image(input_image, channels="BGR", width="stretch")
    if is_binary:
        cols[1].markdown(f"**binaryImage** — ThresholdBinaryInv({thresh}, 255)")
        cols[1].image(binary_image, width="stretch")
    cols[-1].markdown(f"**imgBoxOutput — {title}**")
    cols[-1].image(output, channels="BGR" if output.ndim == 3 else "RGB", width="stretch",
                   caption=f"kernel {shape} {ksize}×{ksize}, iterations = {int(iterations)}")
    st.caption(EXPLAIN[base_op])

# ---------- File > Save ----------
with save_slot:
    # >>> CODE: File > Save: ບັນທຶກຮູບ imgBoxOutput (jpg / png / bmp)
    shown = output if output is not None else input_image
    fmt = st.selectbox("ຮູບແບບ file", ["jpg", "png", "bmp"])
    st.download_button("Save", data=cv2.imencode("." + fmt, shown)[1].tobytes(),
                       file_name=f"{op or 'input'}.{fmt}", mime=f"image/{'jpeg' if fmt == 'jpg' else fmt}",
                       width="stretch", on_click=lambda: st.toast("Saving Complete"))
    # <<< CODE
