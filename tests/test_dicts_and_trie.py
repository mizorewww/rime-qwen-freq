import struct
import tempfile
import unittest
from pathlib import Path

from rime_qwen_freq.dicts import (
    analyze_dict,
    extract_payload,
    read_uint32_npy,
    rewrite_dict_file,
)


def write_sample_dict(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "---\nname: sample\nversion: '1.0'\nsort: by_weight\n...\n"
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


class TestRimeDicts(unittest.TestCase):
    def test_auto_classification_and_extract_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            rime_dir = Path(tmp)
            # 1. Empirical anchor dict (>=1000 rows, >=100 unique weights)
            anchor_rows = [f"词条{i}\tci tiao\t{(i % 250) + 1}" for i in range(1200)]
            write_sample_dict(rime_dir / "dicts" / "base.dict.yaml", anchor_rows)

            # 2. Constant placeholder dict (all 60000)
            wiki_rows = [f"百科{i}\tbai ke\t60000" for i in range(80)]
            wiki_path = rime_dir / "dicts" / "wiki.dict.yaml"
            write_sample_dict(wiki_path, wiki_rows)

            # 3. Unweighted dict (missing 3rd column)
            mixed_rows = ["5G\t5g", "3D\t3d"]
            mixed_path = rime_dir / "dicts" / "mixed.dict.yaml"
            write_sample_dict(mixed_path, mixed_rows)

            # 4. Special typo dict (must be skipped automatically)
            others_path = rime_dir / "dicts" / "rime_ice.others.dict.yaml"
            write_sample_dict(others_path, ["阿房宫\ta fang gong", "空落落\tkong lao lao"])

            self.assertEqual(analyze_dict(rime_dir, rime_dir / "dicts" / "base.dict.yaml").role, "anchor")
            self.assertEqual(analyze_dict(rime_dir, wiki_path).role, "needs-weights")
            self.assertEqual(analyze_dict(rime_dir, mixed_path).role, "needs-weights")
            self.assertEqual(analyze_dict(rime_dir, others_path).role, "skip-special")

            out_payload = rime_dir / "out" / "payload.json.gz"
            payload = extract_payload(rime_dir, out_payload, cn_anchors_max=500)
            self.assertIn("dicts/wiki.dict.yaml", payload["cn_targets"])
            self.assertIn("dicts/mixed.dict.yaml", payload["cn_targets"])
            self.assertNotIn("dicts/rime_ice.others.dict.yaml", payload["cn_targets"])
            self.assertEqual(len(payload["cn_anchors"]), 500)

            tot, upd = rewrite_dict_file(mixed_path, {"5G": 2301, "3D": 4338}, fallback_weight=10)
            self.assertEqual((tot, upd), (2, 2))
            text = mixed_path.read_text(encoding="utf-8")
            self.assertIn("5G\t5g\t2301", text)
            self.assertIn("3D\t3d\t4338", text)

    def test_stdlib_uint32_npy_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "arr.npy"
            descr = "{'descr': '<u4', 'fortran_order': False, 'shape': (4,), }"
            pad_len = (64 - ((10 + len(descr) + 1) % 64)) % 64
            hdr_bytes = (descr + (" " * pad_len) + "\n").encode("latin1")
            raw = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr_bytes)) + hdr_bytes
            raw += struct.pack("<IIII", 3, 374, 14714, 300000)
            p.write_bytes(raw)

            arr = read_uint32_npy(p)
            self.assertEqual(list(arr), [3, 374, 14714, 300000])


if __name__ == "__main__":
    unittest.main()
