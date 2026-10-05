import unittest
from meridian_core.spatial import SpatialEngine,SpatialPoint
from meridian_core.baseline import BaselineEngine
from meridian_core.gwo import GreyWolfOptimizer
class MeridianCoreTests(unittest.TestCase):
 def test_distance_zero(self): self.assertAlmostEqual(SpatialEngine.distance(SpatialPoint("a",6,-75),SpatialPoint("b",6,-75)),0)
 def test_change(self): self.assertGreater(BaselineEngine.detect([10,10,10],12).delta,0)
 def test_gwo_reproducible(self):
  f=lambda x:sum(v*v for v in x);a=GreyWolfOptimizer(f,[(-5,5),(-5,5)],seed=7,iterations=20).run();b=GreyWolfOptimizer(f,[(-5,5),(-5,5)],seed=7,iterations=20).run();self.assertEqual(a.position,b.position)
if __name__=="__main__":unittest.main()
