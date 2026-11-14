"""
ປຸ່ມ "ສະແດງ code ສຳຄັນ" ສຳລັບແຕ່ລະບົດຝຶກຫັດ.

ວິທີໝາຍ code ທີ່ຕ້ອງການສະແດງໃນ file ບົດຝຶກຫັດ:
    ເປີດ block ດ້ວຍ comment  ">>> CODE: <ຫົວຂໍ້>"  (ຂຶ້ນຕົ້ນດ້ວຍ #)
    ປິດ block ດ້ວຍ comment   "<<< CODE"           (ຂຶ້ນຕົ້ນດ້ວຍ #)
code ຈະຖືກອ່ານຈາກ file ແທ້ -> ສິ່ງທີ່ສະແດງກົງກັບ code ທີ່ run ສະເໝີ.
"""
import re
import textwrap
from pathlib import Path

import streamlit as st

_BLOCK = re.compile(r"^[ \t]*# >>> CODE: *(.+?)[ \t]*\n(.*?)^[ \t]*# <<< CODE", re.M | re.S)


def extract_blocks(path: str) -> list[tuple[str, str]]:
    src = Path(path).read_text(encoding="utf-8")
    return [(title, textwrap.dedent(body).strip("\n")) for title, body in _BLOCK.findall(src)]


def show_code_button(path: str) -> None:
    """ສະແດງປຸ່ມ ສະແດງ/ເຊື່ອງ code ສຳຄັນ ຂອງ file `path`."""
    key = "show_code_" + Path(path).stem
    shown = st.session_state.setdefault(key, False)

    def toggle():
        st.session_state[key] = not st.session_state[key]

    st.button("🙈 ເຊື່ອງ code" if shown else "📄 ສະແດງ code ສຳຄັນ", on_click=toggle, key=key + "_btn")
    if shown:
        with st.container(border=True):
            for i, (title, code) in enumerate(extract_blocks(path), 1):
                st.markdown(f"**{i}. {title}**")
                st.code(code, language="python", line_numbers=True)
