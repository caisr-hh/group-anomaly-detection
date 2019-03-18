from cosmo.conformal import pvalue, martingale, Strangeness
from cosmo.utils import InputValidationError, NotFittedError
import unittest, numpy as np

class TestConformal(unittest.TestCase):
    def test_pvalue_general(self):
        self.assertEqual(pvalue(4, [1,2,7,9,11]), 3/5)
        
    def test_pvalue_special(self):
        self.assertEqual(pvalue(0, [1,2,7,9,11]), 1)
        self.assertEqual(pvalue(20, [1,2,7,9,11]), 0)
    
    def test_pvalue_empty(self):
        with self.assertRaises(InputValidationError):
            pvalue(4, [])
        
    def test_martingale_general(self):
        self.assertEqual(martingale([0.2, 0.9, 0.1]), 0.3/1.5)
        self.assertEqual(martingale([0.8, 1, 0.1]), 0)
        self.assertEqual(martingale([0.2, 0.1]), 0.7)
        
    def test_martingale_empty(self):
        with self.assertRaises(InputValidationError):
            martingale([])
        
    def test_martingale_range(self):
        with self.assertRaises(InputValidationError):
            martingale([8, 1, 0.2, 2])
    
    def test_strangeness_median_general(self):
        strg = Strangeness(measure = "median", k = 10)
        self.assertEqual(strg.h.med, None)
        self.assertFalse(strg.is_fitted())
        
    def test_strangeness_knn_general(self):
        strg = Strangeness(measure = "knn", k = 10)
        self.assertEqual(strg.h.X, None)
        self.assertFalse(strg.is_fitted())
        
    def test_strangeness_knn_wrong_k(self):
        with self.assertRaises(InputValidationError):
            strg = Strangeness(measure = "knn", k = 0)
        
    def test_strangeness_wrong_measure(self):
        with self.assertRaises(InputValidationError):
            strg = Strangeness(measure = "foo", k = 10)
            
    def test_strangeness_median_not_fitted(self):
        with self.assertRaises(NotFittedError):
            strg = Strangeness(measure = "median", k = 10)
            strg.get([1,2,3])
            
    def test_strangeness_knn_not_fitted(self):
        with self.assertRaises(NotFittedError):
            strg = Strangeness(measure = "knn", k = 10)
            strg.get([1,2,3])
            
    def test_strangeness_median_fit(self):
        strg = Strangeness(measure = "median", k = 10)
        X = np.array([[1,2,3], [4,5,6], [7,8,9]])
        strg.fit(X)
        self.assertTrue(strg.h.med is not None)
        self.assertTrue(np.allclose(strg.h.med, [4,5,6]))
        self.assertEqual(strg.get([1,2,3]), 27**0.5)
        
    def test_strangeness_knn_fit(self):
        strg = Strangeness(measure = "knn", k = 2)
        X = np.array([[1,2,3], [4,5,6], [7,8,9]])
        strg.fit(X)
        self.assertTrue(strg.h.X is not None)
        self.assertEqual(strg.get([7,8,9]), 27**0.5 / 2)
        
    def test_strangeness_fit_input_wrong(self):
        strg = Strangeness(measure = "median", k = 10)
        
        with self.assertRaises(InputValidationError):
            strg.fit([])
            
        with self.assertRaises(InputValidationError):
            strg.fit("foo")
            
    def test_strangeness_predict_input_wrong(self):
        strg = Strangeness(measure = "median", k = 10)
        X = [[1,2,3], [4,5,6], [7,8,9]]
        strg.fit(X)
        
        with self.assertRaises(InputValidationError):
            strg.get([])
            
        with self.assertRaises(InputValidationError):
            strg.get("foo")
        
if __name__ == '__main__':
    unittest.main()
    