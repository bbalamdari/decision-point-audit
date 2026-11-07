"""The bug and the fix of mask_example/mask.py on the toy trajectory: in "mask" the continue decision after a tool
response is not a label while the final answer is; in "mask_open" both are labels."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mask_example"))
import mask  # noqa: E402


class MaskExample(unittest.TestCase):
    def points(self, mode):
        return mask.next_after_tool_response(mask.TOY, mask.trained_char_spans(mask.TOY, mode))

    def test_bug_supervises_only_stop(self):
        (cont, cont_trained), (stop, stop_trained) = self.points("mask")
        self.assertTrue(cont.startswith("<|channel>"))
        self.assertTrue(stop.startswith("Fixed"))
        self.assertFalse(cont_trained)
        self.assertTrue(stop_trained)

    def test_fix_supervises_both(self):
        self.assertTrue(all(trained for _, trained in self.points("mask_open")))

    def test_fix_keeps_thought_text_out_of_the_loss(self):
        spans = mask.trained_char_spans(mask.TOY, "mask_open")
        i = mask.TOY.index("Edit a.py.")
        self.assertFalse(any(a <= i < b for a, b in spans))


if __name__ == "__main__":
    unittest.main()
