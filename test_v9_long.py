import unittest

from run_v9_long import coadapt_eta


class V9LongScheduleTests(unittest.TestCase):
    def test_coadaptation_rate_halves_in_final_five_epochs(self):
        self.assertEqual(coadapt_eta(1), 0.0001)
        self.assertEqual(coadapt_eta(15), 0.0001)
        self.assertEqual(coadapt_eta(16), 0.00005)
        self.assertEqual(coadapt_eta(17), 0.000025)
        self.assertEqual(coadapt_eta(20), 0.000003125)


if __name__ == "__main__":
    unittest.main()
