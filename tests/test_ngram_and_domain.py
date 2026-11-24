from __future__ import annotations

import inspect
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rime_qwen_freq.bench import _populate_isolated_user_dir, resolve_bench_assets_dir
from rime_qwen_freq.ngram import (
    _forward_last_logits,
    canonical_fuzzy_syllable,
    distill_ngrams_from_lexicon,
    eos_to_rear_scale,
    pmi_to_octagram_scale,
)
from rime_qwen_freq.scorer import bayesian_domain_log10_tilt, compute_domain_lift


class TestNgramAndDomainMath(unittest.TestCase):
    def test_canonical_fuzzy_syllable(self):
        self.assertEqual(canonical_fuzzy_syllable("zhāng"), "zan")
        self.assertEqual(canonical_fuzzy_syllable("chéng"), "cen")
        self.assertEqual(canonical_fuzzy_syllable("shī"), "si")
        self.assertEqual(canonical_fuzzy_syllable("xíng"), "xin")
        self.assertEqual(canonical_fuzzy_syllable("lüè"), "lve")
        # Verify nasal tone vowels ń / ň / ǹ map to 'n' (not 'e')
        self.assertEqual(canonical_fuzzy_syllable("ń"), "n")
        self.assertEqual(canonical_fuzzy_syllable("ńg"), "ng")
        self.assertEqual(canonical_fuzzy_syllable("ň"), "n")
        self.assertEqual(canonical_fuzzy_syllable("ǹ"), "n")

    def test_distill_min_pmi_default_matches_cli(self):
        sig = inspect.signature(distill_ngrams_from_lexicon)
        self.assertAlmostEqual(sig.parameters["min_pmi"].default, 2.35, places=5)

    def test_forward_last_logits_fallback_without_backbone(self):
        class DummyOut:
            def __init__(self, val):
                self.logits = val

        class DummySlice:
            def float(self):
                return "ok_float_logits"

        class DummyLogits:
            def __getitem__(self, item):
                return DummySlice()

        class DummyModel:
            def __call__(self, input_ids, use_cache=False):
                return DummyOut(DummyLogits())

        self.assertEqual(_forward_last_logits(DummyModel(), None, None, [1, 2]), "ok_float_logits")

    def test_pmi_to_octagram_scale_monotonic_and_bounded(self):
        low = pmi_to_octagram_scale(cond_lp=-7.0, pmi=1.8, is_domain=False)
        high = pmi_to_octagram_scale(cond_lp=-2.5, pmi=5.5, is_domain=False)
        dom = pmi_to_octagram_scale(cond_lp=-2.5, pmi=5.5, is_domain=True)
        self.assertGreaterEqual(low, 154_000)
        self.assertLess(low, high)
        self.assertLessEqual(high, dom)
        self.assertLessEqual(dom, 232_000)

    def test_eos_to_rear_scale(self):
        weak = eos_to_rear_scale(-2.8)
        strong = eos_to_rear_scale(-0.8)
        self.assertGreaterEqual(weak, 196_000)
        self.assertGreater(strong, weak)
        self.assertLessEqual(strong, 232_000)

    def test_bayesian_domain_lift(self):
        lift = compute_domain_lift([-4.0, -8.0], [-7.5, -7.8])
        self.assertAlmostEqual(lift[0], 3.5, places=5)
        self.assertAlmostEqual(lift[1], -0.2, places=5)
        tilt = bayesian_domain_log10_tilt(lift, threshold=1.3, slope=0.135, max_boost=0.54)
        self.assertGreater(tilt[0], 0.25)
        self.assertEqual(tilt[1], 0.0)

    def test_bench_assets_synced_and_userdb_isolated(self):
        repo_root = Path(__file__).resolve().parents[1]
        bundled = resolve_bench_assets_dir()
        for fn in ("rime_bench.cc", "rime_api.h", "bench_cases.tsv"):
            self.assertEqual(
                (repo_root / "bench" / fn).read_bytes(),
                (bundled / fn).read_bytes(),
                f"bench/{fn} out of sync with src/rime_qwen_freq/_bench/{fn}",
            )

        with tempfile.TemporaryDirectory() as src_tmp, tempfile.TemporaryDirectory() as dst_tmp:
            src_p, dst_p = Path(src_tmp), Path(dst_tmp)
            (src_p / "rime_mint.schema.yaml").write_text("schema:\n  schema_id: rime_mint\n", encoding="utf-8")
            udb = src_p / "rime_mint.userdb"
            udb.mkdir()
            (udb / "LOCK").write_text("locked", encoding="utf-8")
            (src_p / "user.yaml").write_text("var: 1\n", encoding="utf-8")

            _populate_isolated_user_dir(src_p, dst_p, include_userdb=False)
            self.assertTrue((dst_p / "rime_mint.schema.yaml").exists())
            self.assertFalse((dst_p / "rime_mint.userdb").exists())
            self.assertFalse((dst_p / "user.yaml").exists())

    def test_gram_tool_build_query_merge_and_unicode_roundtrip(self):
        cxx = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
        repo_root = Path(__file__).resolve().parents[1]
        gram_cc = repo_root / "tools" / "gram_tool.cc"
        if not cxx or not gram_cc.is_file():
            self.skipTest("C++ compiler or tools/gram_tool.cc not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            bin_p = tmp_p / "gram_tool"
            subprocess.run([cxx, "-O2", "-std=c++17", str(gram_cc), "-o", str(bin_p)], check=True)

            # Include non-[U+4000, U+9FFF] codepoints (U+00B7 ·, U+3007 〇, U+FF0C ，, U+8000 耀, U+3400 㐀, U+20000 𠀀) + CRLF
            base_tsv = tmp_p / "base.tsv"
            base_tsv.write_bytes(
                (
                    "反向传播\t184206\r\n"
                    "方向传播\t177727\r\n"
                    "零拷贝\t136414\r\n"
                    "住在里屋\t180077\r\n"
                    "卡尔·马克思\t191000\r\n"
                    "二〇〇八年\t188000\r\n"
                    "你好，世界\t176000\r\n"
                    "荣耀时刻\t179000\r\n"
                    "㐀𠀀测试\t172000\r\n"
                ).encode("utf-8")
            )
            base_gram = tmp_p / "base.gram"
            subprocess.run([str(bin_p), "build", str(base_tsv), str(base_gram)], check=True)

            blk = tmp_p / "blacklist.txt"
            blk.write_bytes("方向传播\r\n".encode("utf-8"))
            add_tsv = tmp_p / "add.tsv"
            add_tsv.write_text("推测解码\t195000\n解决问题$\t215000\n", encoding="utf-8")
            merged_gram = tmp_p / "merged.gram"
            subprocess.run(
                [str(bin_p), "merge", str(base_gram), str(blk), str(add_tsv), "140000", str(merged_gram)],
                check=True,
            )

            q_txt = tmp_p / "q.txt"
            q_txt.write_text(
                "反向传播\n方向传播\n零拷贝\n推测解码\n解决问题$\n住在里屋\n"
                "卡尔·马克思\n二〇〇八年\n你好，世界\n荣耀时刻\n㐀𠀀测试\n",
                encoding="utf-8",
            )
            out_tsv = tmp_p / "out.tsv"
            subprocess.run([str(bin_p), "query", str(merged_gram), str(q_txt), str(out_tsv)], check=True)

            got = dict(line.split("\t") for line in out_tsv.read_text(encoding="utf-8").splitlines())
            self.assertEqual(int(got["反向传播"]), 184206)
            self.assertEqual(int(got["方向传播"]), -1)
            self.assertEqual(int(got["零拷贝"]), -1)
            self.assertEqual(int(got["推测解码"]), 195000)
            self.assertEqual(int(got["解决问题$"]), 215000)
            self.assertEqual(int(got["住在里屋"]), 180077)
            self.assertEqual(int(got["卡尔·马克思"]), 191000)
            self.assertEqual(int(got["二〇〇八年"]), 188000)
            self.assertEqual(int(got["你好，世界"]), 176000)
            self.assertEqual(int(got["荣耀时刻"]), 179000)
            self.assertEqual(int(got["㐀𠀀测试"]), 172000)

            # Verify scan_typos decodes non-[U+4000..U+9FFF] codepoints (·, 〇, 耀, 𠀀) back to exact UTF-8
            fpy_tsv = tmp_p / "fpy.tsv"
            fpy_tsv.write_text(
                "二\ter\n〇\tlin\n零\tlin\n八\tba\n年\tnian\n"
                "㐀\tqiu\n𠀀\the\n测\tce\n试\tsi\n求\tqiu\n合\the\n",
                encoding="utf-8",
            )
            base_words = tmp_p / "base_words.txt"
            base_words.write_text("二零零八年\n求合测试\n", encoding="utf-8")
            suspects = tmp_p / "suspects.tsv"
            subprocess.run(
                [str(bin_p), "scan_typos", str(merged_gram), str(fpy_tsv), str(base_words), str(suspects)],
                check=True,
            )
            sus_lines = suspects.read_text(encoding="utf-8").splitlines()
            sus_map = {ln.split("\t")[0]: ln.split("\t")[2] for ln in sus_lines}
            self.assertEqual(sus_map.get("二〇〇八年"), "二零零八年")
            self.assertEqual(sus_map.get("㐀𠀀测试"), "求合测试")


if __name__ == "__main__":
    unittest.main()
