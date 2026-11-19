"""Run with python3 -m unittest discover -s tests -p test_pipeline_split.py."""

import unittest

from pipeline_split import layout_string, parse_stage_layers, stage_ranges, validate_stage_layers


class PipelineSplitTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_stage_layers("7,9,9,7"), (7, 9, 9, 7))
        self.assertIsNone(parse_stage_layers(None))
        self.assertIsNone(parse_stage_layers(""))
        with self.assertRaises(ValueError):
            parse_stage_layers("7,x")

    def test_validate(self):
        self.assertEqual(validate_stage_layers([7, 9, 9, 7], 32, 4), (7, 9, 9, 7))
        self.assertEqual(validate_stage_layers([3, 5, 5, 3], 16, 2, 2), (3, 5, 5, 3))
        for counts, layers, pp, vpp in (
            ((7, 9, 9), 25, 4, None),    # wrong length
            ((8, 8, 8, 9), 32, 4, None),  # wrong sum
            ((0, 16), 16, 2, None),       # empty stage
            ((8, 8), 16, 2, 2),           # VPP needs PP * VPP entries
        ):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                validate_stage_layers(counts, layers, pp, vpp)

    def test_layout_string(self):
        self.assertEqual(layout_string((7, 9, 9, 7)), "Et*7|t*9|t*9|t*7L")
        self.assertEqual(layout_string((1, 3), mtp_num_layers=1), "Et*1|t*3mL")
        self.assertEqual(layout_string((4,)), "Et*4L")

    def test_stage_ranges(self):
        self.assertEqual(stage_ranges((7, 9, 9, 7)), ((0, 7), (7, 9), (16, 9), (25, 7)))


if __name__ == "__main__":
    unittest.main()
