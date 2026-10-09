#!/usr/bin/env python3
"""Small, deterministic regression tests for the QFS-DMAE certification safety patch.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_rescue_safety.py' -v
These are lightweight CPU tests, not ImageNet certification or an end-to-end
reproduction. They intentionally do NOT certify the colored QWT mixture.
"""
import unittest
import torch

from util.noise import add_noise
from util.quadatasetgpu import QuaternionWavelet, QuaternionWaveletNoise
from util.smooth import Smooth


class CertificationSafetyTest(unittest.TestCase):
    def test_qwt_certification_fails_closed(self):
        # A base classifier can be any nn.Module: this test must fail before
        # an invalid Cohen radius can be reported for a non-isotropic noise law.
        with self.assertRaisesRegex(ValueError, "QWT-corrupted inference"):
            Smooth(torch.nn.Identity(), num_classes=2, sigma=0.5,
                   use_quaternion_noise=True, device="cpu")

    def test_gaussian_mode_keeps_canonical_formula(self):
        s = Smooth(torch.nn.Identity(), num_classes=2, sigma=0.5,
                   use_quaternion_noise=False, device="cpu")
        self.assertAlmostEqual(s.sigma_pix, 0.5)
        self.assertFalse(s.use_qwt)

    def test_haar_analysis_synthesis_identity(self):
        torch.manual_seed(41)
        images = torch.randn(2, 8, 8, 3)
        transform = QuaternionWavelet(device="cpu")
        reconstruction = transform.reconstruct(transform.decompose(images, levels=1))
        torch.testing.assert_close(reconstruction, images, atol=1e-5, rtol=1e-5)

    def test_all_32_finite_orthogonal_transform_inverses(self):
        x = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
        for flip in range(2):
            for k in range(4):
                for dx in range(2):
                    for dy in range(2):
                        y = torch.flip(x, dims=(3,)) if flip else x
                        y = torch.rot90(y, k, dims=(2, 3))
                        y = torch.roll(y, shifts=(dy, dx), dims=(2, 3))
                        code = torch.tensor([k + 4 * flip + 8 * (dx + 2 * dy)])
                        recovered = QuaternionWaveletNoise._invert_transform(y, code)
                        torch.testing.assert_close(recovered, x, atol=0, rtol=0)

    def test_zero_noise_is_identity(self):
        torch.manual_seed(10)
        x = torch.randn(2, 3, 8, 8)
        y = QuaternionWaveletNoise.apply_noise(
            x, sigma=0.0, levels=1, ratio=3.0, device="cpu")
        torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)

    def test_qwt_training_sigma_is_per_channel(self):
        # This is a variance sanity check ONLY; it is not a robustness proof.
        torch.manual_seed(37)
        x = torch.zeros(80, 3, 8, 8)
        y = add_noise(x, sigma=0.5, use_quaternion_noise=True,
                      levels=1, ratio=3.0, device="cpu")
        self.assertLess(abs(y.std(unbiased=False).item() - 0.5), 0.08)

    def test_multilevel_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one QWT level"):
            QuaternionWaveletNoise(0.5, levels=2, device="cpu")


if __name__ == "__main__":
    unittest.main()
