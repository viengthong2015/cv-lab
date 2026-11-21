"""ໜ້າຫຼັກ: ລາຍການບົດຝຶກຫັດ ວາງເປັນ patch ເທິງແຜ່ນທຽບສີ ColorChecker."""
import sys
from html import escape
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from ui import EXERCISES, page_header  # noqa: E402

st.set_page_config(page_title="Computer Vision Lab", layout="wide")
page_header(None, "ຫ້ອງທົດລອງ Computer Vision", "ບົດຝຶກຫັດ EmguCV (C#) ທີ່ແປງມາເປັນ Python + OpenCV")

patches = "".join(
    f'<a class="cv-patch" href="{ex["path"]}" target="_self" title="{escape(ex["title"])}" '
    f'style="--patch:{ex["patch"]};--fg:{ex["fg"]}" aria-label="ບົດຝຶກຫັດ {n}: {escape(ex["title"])}">'
    f'<span class="cv-patch-num">{n}</span>'
    f'<span class="cv-patch-name">{escape(ex["short"])}</span>'
    f'<span class="cv-patch-en">{escape(ex["en_short"])}</span></a>'
    for n, ex in EXERCISES.items()
)

st.html(f"""
<style>
  .cv-lede {{ max-width: 44rem; margin: 1.25rem 0 1.5rem; font-size: 1rem; line-height: 1.6; color: #343434; }}
  .cv-board-wrap {{ container-type: inline-size; }}
  .cv-board {{ background: #262626; border-radius: 0.6rem; padding: 12px;
               display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }}
  .cv-patch {{ min-height: 8.5rem; border-radius: 0.3rem; padding: 0.7rem 0.75rem;
               background: var(--patch); color: var(--fg); text-decoration: none;
               display: flex; flex-direction: column; outline-offset: 3px; }}
  .cv-patch:hover, .cv-patch:focus-visible {{ outline: 3px solid #F3F3F2; }}
  .cv-patch-num {{ font-size: 2rem; font-weight: 600; line-height: 1; font-variant-numeric: tabular-nums; }}
  .cv-patch-name {{ margin-top: auto; font-size: 0.95rem; font-weight: 600; line-height: 1.4; }}
  .cv-patch-en {{ font-size: 0.8rem; line-height: 1.35; }}
  @container (max-width: 720px) {{ .cv-board {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
</style>
<p class="cv-lede">ເລືອກບົດຝຶກຫັດຈາກແຜ່ນລຸ່ມນີ້ ຫຼື ຈາກເມນູດ້ານຊ້າຍ.
ທຸກໜ້າມີປຸ່ມ <strong>ສະແດງ code ສຳຄັນ</strong> ເພື່ອເບິ່ງ code Python ທຽບກັບ code C# ຂອງບົດຮຽນ.</p>
<div class="cv-board-wrap"><nav class="cv-board" aria-label="ລາຍການບົດຝຶກຫັດ">{patches}</nav></div>
""")
