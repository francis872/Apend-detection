import unittest
from fastapi.testclient import TestClient
from meridian_api.app import app
class MeridianApiTests(unittest.TestCase):
 def setUp(self): self.client=TestClient(app)
 def test_health(self): self.assertEqual(self.client.get("/api/v1/health").status_code,200)
 def test_property_intelligence_uses_real_context(self):
  body={"propertyId":"p1","latitude":6.25,"longitude":-75.56,"radiusKm":5,"contextNodes":[{"id":"p2","latitude":6.251,"longitude":-75.561}],"history":[]}
  data=self.client.post("/api/v1/properties/intelligence",json=body).json()
  self.assertEqual(data["nearbyCount"],1);self.assertIsNone(data["riskScore"]);self.assertIsNone(data["forecastNext"])
if __name__=="__main__":unittest.main()
