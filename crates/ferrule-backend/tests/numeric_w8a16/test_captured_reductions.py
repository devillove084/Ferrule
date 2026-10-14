"""No-GPU regression tests for the real-operand reduction experiment."""
import unittest
import numpy as np
import prepare_captured_triangle as capture


class CapturedReductionTests(unittest.TestCase):
    def test_rne_signed_ties(self):
        words=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000],dtype=np.uint32)
        expected=np.array([0x3f800000,0x3f820000,0xbf800000,0xbf820000],dtype=np.uint32)
        np.testing.assert_array_equal(capture.rne(words.view(np.float32)).view(np.uint32),expected)

    def test_serial_cancellation_is_not_a_better_reference(self):
        x=np.ones((1,128),dtype=np.float32);w=np.zeros((1,128),dtype=np.float32)
        w[0,[0,4,8]]=[2.**24,1.,-2.**24]
        results,truth,norms=capture.reductions(x,w)
        self.assertEqual(truth.item(),1.)
        self.assertEqual(results['serial'].item(),0.)
        self.assertEqual(results['vector_replay'].item(),1.)
        self.assertEqual(capture.metrics(results['serial'],truth,norms)['bound_failures'],0)

    def test_two_bank_and_pairwise_retain_f32_not_f64_outputs(self):
        x=np.ones((1,256),dtype=np.float32);w=np.zeros((1,256),dtype=np.float32)
        w[0,[0,128,129]]=[2.**24,1.,-2.**24]
        results,truth,_=capture.reductions(x,w)
        self.assertEqual(truth.item(),1.)
        for value in results.values(): self.assertEqual(value.dtype,np.float32)
        self.assertEqual(results['serial'].item(),0.)

    def test_identity_multiple_rows_and_tiles(self):
        x=np.arange(768,dtype=np.float32).reshape(3,256)
        x=capture.rne(x);w=np.eye(256,dtype=np.float32)[:5]
        result,truth,_=capture.reductions(x,w)
        for value in result.values(): np.testing.assert_array_equal(value,x[:,:5])
        np.testing.assert_array_equal(truth,x[:,:5])

    def test_reject_non_bf16_and_unrepresentable_products(self):
        x=np.ones((1,128),dtype=np.float32);w=x.copy();x[0,0]=1.001
        with self.assertRaises(ValueError):capture.reductions(x,w)
        x=np.full((1,128),2.**-126,dtype=np.float32)
        w=np.full((1,128),2.**-126,dtype=np.float32)
        with self.assertRaises(ValueError):capture.reductions(x,w)

    def test_balanced_tree_pads_non_power_of_two(self):
        values=np.array([[1.,2.,4.]],dtype=np.float32)
        np.testing.assert_array_equal(capture.tree_reduce(values,-1),[7.])


if __name__=='__main__':unittest.main(verbosity=2)
