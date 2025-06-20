import unittest
from src.core.archangel_god import QARCHANGEL
import numpy as np

class TestQARCHANGEL(unittest.TestCase):
    def test_forward_pass(self):
        model = QARCHANGEL()
        text = "chest pain"
        image = np.random.rand(224, 224, 3)
        audio = np.random.rand(16000)
        outputs = model.forward(text, image, audio)
        self.assertIn("diagnosis", outputs)
        self.assertIn("treatment", outputs)
        self.assertIn("severity", outputs)
        self.assertIn("emergency", outputs)

if __name__ == "__main__":
    unittest.main()