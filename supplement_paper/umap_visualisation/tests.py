import unittest
import numpy as np
import umap_vis


class TestUmapVis(unittest.TestCase):

    def test_reshape_data(self):
        inp_read_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        inp_read_2 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        inp = np.array([inp_read_1, inp_read_2])
        out_read_1 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        out_read_2 = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        out = np.array([out_read_1, out_read_2])
        self.assertEqual(umap_vis.reshape_data(inp).tolist(), out.tolist())
        self.assertEqual(np.testing.assert_array_equal(umap_vis.reshape_data(inp), out), None)


if __name__ == '__main__':
    unittest.main()
