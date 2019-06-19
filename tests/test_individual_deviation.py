from grand import IndividualAnomalyInductive
from grand.utils import InputValidationError, NotFittedError, DeviationContext
import numpy as np
import unittest

class TestIndividualDeviation(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(InputValidationError):
            IndividualAnomalyInductive(w_martingale=0, non_conformity="median", k=20, dev_threshold=0.6)
        
        with self.assertRaises(InputValidationError):
            IndividualAnomalyInductive(w_martingale=-1, non_conformity="median", k=20, dev_threshold=0.6)
        
        with self.assertRaises(InputValidationError):
            IndividualAnomalyInductive(w_martingale=15, non_conformity="foo", k=20, dev_threshold=0.6)
        
        with self.assertRaises(InputValidationError):
            IndividualAnomalyInductive(w_martingale=15, non_conformity="knn", k=0, dev_threshold=0.6)
        
        with self.assertRaises(InputValidationError):
            IndividualAnomalyInductive(w_martingale=15, non_conformity="knn", k=-1, dev_threshold=0.6)
        
        with self.assertRaises(InputValidationError):
            IndividualAnomalyInductive(w_martingale=15, non_conformity="median", k=20, dev_threshold=-1)
        
        with self.assertRaises(InputValidationError):
            IndividualAnomalyInductive(w_martingale=15, non_conformity="median", k=20, dev_threshold=2)
    
    def test_fit_input_empty(self):
        indev = IndividualAnomalyInductive(w_martingale=15, non_conformity="median", k=20, dev_threshold=0.6)
        with self.assertRaises(InputValidationError):
            indev.fit([])
        
    def test_fit_median(self):
        indev = IndividualAnomalyInductive(w_martingale=15, non_conformity="median", k=20, dev_threshold=0.6)
        indev.fit(np.array([[1,2,3], [4,5,6], [7,8,9]]))
        expected = [3**(3/2), 0, 3**(3/2)]
        self.assertEqual(indev.scores, expected)
        self.assertTrue(np.allclose(indev.scores, expected))
        
    def test_fit_knn_k20(self):
        indev = IndividualAnomalyInductive(w_martingale=15, non_conformity="knn", k=20, dev_threshold=0.6)
        indev.fit(np.array([[1,2,3], [4,5,6], [7,8,9]]))
        expected = [ (0 + 3**(3/2) + (3*6*6)**0.5) / 3, (3**(3/2) + 0 + 3**(3/2)) / 3, ((3*6*6)**0.5 + 3**(3/2) + 0) / 3 ]
        self.assertEqual(indev.scores, expected)
        self.assertTrue(np.allclose(indev.scores, expected))
        
    def test_fit_knn_k2(self):
        indev = IndividualAnomalyInductive(w_martingale=15, non_conformity="knn", k=2, dev_threshold=0.6)
        indev.fit(np.array([[1,2,3], [4,5,6], [7,8,9]]))
        expected = [ (0 + 3**(3/2)) / 2, (3**(3/2) + 0) / 2, (3**(3/2) + 0) / 2 ]
        self.assertEqual(indev.scores, expected)
        self.assertTrue(np.allclose(indev.scores, expected))
        
    def test_predict_median_w15(self):
        indev = IndividualAnomalyInductive(w_martingale=15, non_conformity="median", k=20, dev_threshold=0.6)
        indev.fit(np.array([[1,2,3], [4,5,6], [7,8,9]]))
        res = indev.predict(None, [1,1,1])
        expected = DeviationContext((3**2+4**2+5**2)**0.5, 0, 0.25, False)
        self.assertEqual(res, expected)
        
    def test_predict_median_w3(self):
        indev = IndividualAnomalyInductive(w_martingale=3, non_conformity="median", k=20, dev_threshold=0.6)
        indev.fit(np.array([[1,2,3], [4,5,6], [7,8,9]]))
        res = indev.predict(None, [1,1,1])
        expected = DeviationContext((3**2+4**2+5**2)**0.5, 0, 1/3, False)
        self.assertEqual(res, expected)
        
    def test_predict_not_fitted(self):
        indev = IndividualAnomalyInductive(w_martingale=15, non_conformity="median", k=20, dev_threshold=0.6)
        with self.assertRaises(NotFittedError):
            indev.predict(None, [1,1,1])
            
    def test_predict_input_wrong(self):
        indev = IndividualAnomalyInductive(w_martingale=15, non_conformity="median", k=20, dev_threshold=0.6)
        indev.fit(np.array([[1,2,3], [4,5,6], [7,8,9]]))
        
        with self.assertRaises(InputValidationError):
            indev.predict(None, [])
            
        with self.assertRaises(InputValidationError):
            indev.predict(None, "foo")
            
if __name__ == '__main__':
    unittest.main()
    