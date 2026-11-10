import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from distillation_loop.student import require_single_gpu, clear_cuda_cache


class GPUSetupTests(unittest.TestCase):
    def test_rejects_multiple_visible_devices_before_training(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
        with self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES=0'):
            require_single_gpu(torch)
        torch.cuda.device_count = lambda: 1
        require_single_gpu(torch)

    def test_cleanup_preserves_original_training_error(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock(side_effect=RuntimeError('illegal memory access'))))
        with self.assertRaisesRegex(ValueError, 'original training failure'):
            try:
                raise ValueError('original training failure')
            finally:
                clear_cuda_cache(torch)
        with self.assertRaisesRegex(RuntimeError, 'illegal memory access'):
            clear_cuda_cache(torch)
