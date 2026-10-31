"""最终困惑度表的计算口径测试。"""
from pathlib import Path
import sys, unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_perplexity_outputs as outputs

class Tests(unittest.TestCase):
    def test_summary_uses_sample_sd_and_signed_improvement(self):
        rows=[]
        for k, vals in [(20, [10.,12.,14.]), (30,[8.,10.,12.])]:
            for seed, value in zip([42,2026,20260830], vals): rows.append({"corpus":"C","K":str(k),"seed":str(seed),"test_perplexity":str(value),"train_perplexity":"5"})
        got=outputs.summarize(rows)
        self.assertEqual(got[0]["std_test_perplexity"], "2.0000000000")
        self.assertEqual(got[1]["absolute_improvement"], "2.0000000000")
        self.assertAlmostEqual(float(got[1]["relative_improvement_percent"]), 100/6)

if __name__ == "__main__": unittest.main()
