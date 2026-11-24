import struct
import tempfile
import unittest
from pathlib import Path

from rime_qwen_freq import __version__
from rime_qwen_freq.cli import build_parser
from rime_qwen_freq.dicts import (
    analyze_dict,
    extract_payload,
    read_float16_npy,
    read_uint32_npy,
    rewrite_dict_file,
)
from rime_qwen_freq.scorer import (
    bayesian_domain_log10_tilt,
    build_prefix_trie,
    compute_domain_lift,
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

            # 4. Special typo & user custom dicts (must be skipped automatically)
            others_path = rime_dir / "dicts" / "rime_ice.others.dict.yaml"
            write_sample_dict(others_path, ["阿房宫\ta fang gong", "空落落\tkong lao lao"])
            custom_path = rime_dir / "dicts" / "custom_simple.dict.yaml"
            write_sample_dict(custom_path, ["自定义\tzi ding yi"])

            self.assertEqual(analyze_dict(rime_dir, rime_dir / "dicts" / "base.dict.yaml").role, "anchor")
            self.assertEqual(analyze_dict(rime_dir, wiki_path).role, "needs-weights")
            self.assertEqual(analyze_dict(rime_dir, mixed_path).role, "needs-weights")
            self.assertEqual(analyze_dict(rime_dir, others_path).role, "skip-special")
            self.assertEqual(analyze_dict(rime_dir, custom_path).role, "skip-special")

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

    def test_stdlib_float16_npy_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "delta.npy"
            descr = "{'descr': '<f2', 'fortran_order': False, 'shape': (3,), }"
            pad_len = (64 - ((10 + len(descr) + 1) % 64)) % 64
            hdr_bytes = (descr + (" " * pad_len) + "\n").encode("latin1")
            raw = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr_bytes)) + hdr_bytes
            raw += struct.pack("<eee", -1.25, 0.0, 4.5)
            p.write_bytes(raw)

            vals = read_float16_npy(p)
            self.assertEqual([round(x, 3) for x in vals], [-1.25, 0.0, 4.5])

    def test_prefix_trie_and_bayesian_domain_lift(self):
        self.assertEqual(__version__, "0.4.0")
        trie = build_prefix_trie([[10, 20, 30], [10, 20, 40], [10, 50], [88, 77], [99]])
        self.assertEqual(len(trie["leaves"]), 2)
        self.assertEqual(trie["num_edges"], 8)

        lifts = compute_domain_lift([-6.0, -9.5, -12.0], [-10.5, -10.5, -10.0])
        self.assertEqual([round(x, 2) for x in lifts], [4.5, 1.0, -2.0])

        tilts = bayesian_domain_log10_tilt(lifts, threshold=1.3, slope=0.135, max_boost=0.54)
        self.assertAlmostEqual(tilts[0], 0.432, places=4)
        self.assertEqual(tilts[1], 0.0)
        self.assertEqual(tilts[2], 0.0)

        capped = bayesian_domain_log10_tilt([8.0], threshold=1.3, slope=0.135, max_boost=0.54)
        self.assertAlmostEqual(capped[0], 0.54, places=4)

    def test_cli_domain_prompt_arguments(self):
        parser = build_parser()
        args = parser.parse_args([
            "score",
            "-i", "in.json.gz",
            "-o", "out.tar.gz",
            "--model", "Qwen/Qwen3-14B-Base",
            "--domain-prompt", "计算机科学与人工智能术语：",
            "--general-prompt", "日常与百科常见词条：",
            "--domain-threshold", "1.5",
            "--domain-max-boost", "0.5",
        ])
        self.assertEqual(args.model, "Qwen/Qwen3-14B-Base")
        self.assertEqual(args.domain_prompt, "计算机科学与人工智能术语：")
        self.assertEqual(args.general_prompt, "日常与百科常见词条：")
        self.assertAlmostEqual(args.domain_threshold, 1.5)
        self.assertAlmostEqual(args.domain_max_boost, 0.5)


if __name__ == "__main__":
    unittest.main()
