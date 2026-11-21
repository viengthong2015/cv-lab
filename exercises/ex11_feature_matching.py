"""
ບົດຝຶກຫັດ 11: ORB and SIFT Feature Matching
C# (EmguCV)                                               ->  Python (OpenCV + Streamlit)
-------------------------------------------------------------------------------------------
imageBox1_Click: Timage (Template)                        ->  st.file_uploader (Template)
imageBox2_Click: Simage (Image Source)                    ->  st.file_uploader (Image Source)
ORB orb = new ORB(numberOfFeatures: 500)                  ->  cv2.ORB_create(nfeatures=500)
SIFT sift = new SIFT()                                    ->  cv2.SIFT_create()
x.DetectAndCompute(img, null, kpts, desc, false)          ->  kpts, desc = x.detectAndCompute(img, None)
new BFMatcher(DistanceType.Hamming)  (ORB)                ->  cv2.BFMatcher(cv2.NORM_HAMMING)
new BFMatcher(DistanceType.L2)       (SIFT)               ->  cv2.BFMatcher(cv2.NORM_L2)
matcher.Add(desc1); matcher.KnnMatch(desc2, matches, k: 2) ->  bf.knnMatch(desc2, desc1, k=2)
Features2DToolbox.VoteForUniqueness(matches, 0.8, mask)   ->  Lowe's ratio test: d1 / d2 ≤ 0.8
Features2DToolbox.DrawMatches(..., MCvScalar(0,255,0), MCvScalar(255,0,0), mask)
                                                          ->  cv2.drawMatchesKnn(..., matchesMask=mask)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from code_viewer import show_code_button  # noqa: E402
from ui import page_header  # noqa: E402

st.set_page_config(page_title="Ex11 - ORB & SIFT", layout="wide")
page_header(11)
show_code_button(__file__)

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]


def read_bgr(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# >>> CODE: btnORBMatch_Click / btnSIFT_Click: Detect -> Compute -> KnnMatch -> Ratio Test -> DrawMatches
@st.cache_data(show_spinner="ກຳລັງຈັບຄູ່ຈຸດພິເສດ...")
def feature_matching(t_image: np.ndarray, s_image: np.ndarray, algorithm: str,
                     n_features: int = 500, ratio: float = 0.8, only_good: bool = False):
    img1, img2 = t_image, s_image                                    # Mat img1 = Timage.Mat; img2 = Simage.Mat

    # 2. ສ້າງ Object ຂອງ ORB ຫຼື SIFT (+ ເລືອກໄລຍະຫ່າງທີ່ເໝາະສົມ)
    if algorithm == "ORB":
        detector = cv2.ORB_create(nfeatures=n_features)              # new ORB(numberOfFeatures: 500)
        norm = cv2.NORM_HAMMING        # ORB descriptor ເປັນ binary -> ຕ້ອງໃຊ້ Hamming ເທົ່ານັ້ນ
    else:
        detector = cv2.SIFT_create()                                 # new SIFT()
        norm = cv2.NORM_L2             # SIFT descriptor ເປັນ float -> ໃຊ້ L2 (Euclidean)

    # 3. ກວດຫາ Keypoints ແລະ ຄິດໄລ່ Descriptor
    kpts1, desc1 = detector.detectAndCompute(img1, None)             # DetectAndCompute(img1, null, kpts1, desc1)
    kpts2, desc2 = detector.detectAndCompute(img2, None)
    if desc1 is None or desc2 is None:
        return None, len(kpts1), len(kpts2), 0

    # 4. ການຈັບຄູ່: Template = train (matcher.Add(desc1)), Source = query (KnnMatch(desc2, k: 2))
    matcher = cv2.BFMatcher(norm)
    matches = matcher.knnMatch(desc2, desc1, k=2)

    # 5. Lowe's Ratio Test (VoteForUniqueness 0.8): ເກັບຄູ່ທີ່ໃກ້ທີ່ສຸດ ຖ້າມັນໃກ້ກວ່າອັນດັບ 2 ຢ່າງຊັດເຈນ
    mask = []
    for m in matches:
        good = len(m) < 2 or m[0].distance <= ratio * m[1].distance
        mask.append([1 if (good and i == 0) else 0 for i in range(len(m))])
    n_good = sum(row[0] for row in mask if row)

    # 6. ສະແດງຜົນ: ເສັ້ນສີຂຽວ = ຄູ່ທີ່ຜ່ານ ratio test, ຈຸດສີຟ້າ = keypoint ທີ່ບໍ່ໄດ້ຈັບຄູ່
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS if only_good else cv2.DrawMatchesFlags_DEFAULT
    result_image = cv2.drawMatchesKnn(img2, kpts2, img1, kpts1, matches, None,
                                      matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                      matchesMask=mask, flags=flags)
    return result_image, len(kpts1), len(kpts2), n_good
# <<< CODE


ss = st.session_state
ss.setdefault("ex11_ids", (None, None))
ss.setdefault("ex11_algo", None)  # ປຸ່ມທີ່ກົດຫຼ້າສຸດ: "ORB" / "SIFT"

# ---------- Template | Image Source ----------
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Template** (imageBox1)")
    # >>> CODE: imageBox1_Click / imageBox2_Click: ເລືອກຮູບ Template ແລະ Image Source
    t_file = st.file_uploader("ເລືອກຮູບ Template", type=IMAGE_TYPES, label_visibility="collapsed", key="ex11_t")
    # <<< CODE
    t_image = read_bgr(t_file) if t_file is not None else None
    if t_image is not None:
        st.image(t_image, channels="BGR", width="content")
with c2:
    st.markdown("**Image Source** (imageBox2)")
    s_file = st.file_uploader("ເລືອກຮູບ Image Source", type=IMAGE_TYPES, label_visibility="collapsed", key="ex11_s")
    s_image = read_bgr(s_file) if s_file is not None else None
    if s_image is not None:
        st.image(s_image, channels="BGR", width="content")

ids = (t_file.file_id if t_file else None, s_file.file_id if s_file else None)
if ss.ex11_ids != ids:  # ປ່ຽນຮູບ -> ລ້າງຜົນເກົ່າ
    ss.ex11_ids, ss.ex11_algo = ids, None
ready = t_image is not None and s_image is not None

# ---------- btnORBMatch | btnSIFT | parameter ----------
b1, b2, b3, _ = st.columns([1.6, 1.6, 2.2, 2.6])
if b1.button("ORB Matching", type="primary", width="stretch", disabled=not ready):
    ss.ex11_algo = "ORB"
if b2.button("SIFT Matching", type="primary", width="stretch", disabled=not ready):
    ss.ex11_algo = "SIFT"
with b3.popover("⚙️ ປັບຄ່າ parameter", width="stretch"):
    st.caption("ຄ່າເລີ່ມຕົ້ນ = ຄ່າໃນ code C#")
    n_features = st.slider("ORB: numberOfFeatures", 100, 5000, 500, step=100)
    ratio = st.slider("Ratio (VoteForUniqueness)", 0.5, 1.0, 0.8, step=0.05,
                      help="ນ້ອຍລົງ = ເຂັ້ມງວດຂຶ້ນ, ຄູ່ໜ້ອຍລົງແຕ່ຖືກຕ້ອງກວ່າ. ນິຍົມໃຊ້ 0.7–0.8")
    only_good = st.toggle("ສະແດງສະເພາະຄູ່ທີ່ຜ່ານ (ບໍ່ແຕ້ມຈຸດສີຟ້າ)", value=False)

# ---------- imageBox3: Matching Result ----------
st.markdown("**Matching Result** (imageBox3)")
if not ready:
    st.info("ເລືອກຮູບ **Template** ແລະ **Image Source** ກ່ອນ.")
    st.stop()
if ss.ex11_algo is None:
    st.info("ກົດ **ORB Matching** ຫຼື **SIFT Matching**.")
    st.stop()

result, n_kp1, n_kp2, n_good = feature_matching(t_image, s_image, ss.ex11_algo, n_features, ratio, only_good)
if result is None:
    st.error(f"{ss.ex11_algo}: ຫາ keypoint ບໍ່ພົບໃນຮູບໃດຮູບໜຶ່ງ (Template {n_kp1}, Source {n_kp2}) "
             "— ລອງໃຊ້ຮູບທີ່ມີລາຍລະອຽດ/ມຸມຫຼາຍກວ່ານີ້.")
    st.stop()

st.image(result, channels="BGR", width="stretch",
         caption=f"{ss.ex11_algo}: keypoints Template = {n_kp1}, Source = {n_kp2} → "
                 f"ຄູ່ທີ່ຜ່ານ ratio test ({ratio:.2f}) = {n_good}. ຊ້າຍ = Image Source, ຂວາ = Template")
if n_kp1 < 30:
    st.warning(f"Template ມີພຽງ {n_kp1} keypoints. ORB ບໍ່ຊອກຫາຈຸດໃກ້ຂອບຮູບ (31 px) ຈຶ່ງຫາໄດ້ໜ້ອຍໃນຮູບນ້ອຍ. "
               "ເນື່ອງຈາກ Source ເປັນຝ່າຍຄົ້ນຫາ (KnnMatch(desc2)), ຫຼາຍຈຸດໃນ Source ອາດຈັບຄູ່ກັບຈຸດດຽວກັນໃນ Template "
               "— ລອງໃຊ້ Template ທີ່ໃຫຍ່ຂຶ້ນ ຫຼື ໃຊ້ SIFT.")
st.caption("ORB: ໄວ, descriptor ເປັນ binary (Hamming). SIFT: ຊ້າກວ່າ ແຕ່ທົນຕໍ່ການປ່ຽນຂະໜາດ ແລະ ການໝຸນໄດ້ດີກວ່າ "
           "(descriptor float, L2).")
st.download_button("Save", data=cv2.imencode(".jpg", result)[1].tobytes(),
                   file_name=f"{ss.ex11_algo.lower()}_matching.jpg", mime="image/jpeg",
                   on_click=lambda: st.toast("Saving Complete"))
