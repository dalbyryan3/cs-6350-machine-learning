import unittest
from decision_tree import DecisionTree

class DecisionTreeTester(unittest.TestCase):
    def setUp(self):
        self.labels = [False,False,True,True,True,False,True,False,True,True,True,True,True,False]
        self.examples = [
            self.be('Sunny','Hot', 'High', 'Weak'),
            self.be('Sunny','Hot', 'High', 'Strong'),
            self.be('Overcast','Hot', 'High', 'Weak'),
            self.be('Rainy','Medium', 'High', 'Weak'),
            self.be('Rainy','Cold', 'Normal', 'Weak'),
            self.be('Rainy','Cold', 'Normal', 'Strong'),
            self.be('Overcast','Cold', 'Normal', 'Strong'),
            self.be('Sunny','Medium', 'High', 'Weak'),
            self.be('Sunny','Cold', 'Normal', 'Weak'),
            self.be('Rainy','Medium', 'Normal', 'Weak'),
            self.be('Sunny','Medium', 'Normal', 'Strong'),
            self.be('Overcast','Medium', 'High', 'Strong'),
            self.be('Overcast','High', 'Normal', 'Weak'),
            self.be('Rainy','Medium', 'High', 'Strong')
        ]
    def be(self, outlook, temperature, humidity, wind):
        # Build example
        return {'Outlook':outlook, 'Temperature':temperature,'Humidity':humidity, 'Wind':wind}
    def test_entropy(self):
        self.assertAlmostEqual(DecisionTree.entropy(self.labels), 0.652, places=3)
    def test_majority_error(self):
        self.assertAlmostEqual(DecisionTree.majority_error(self.labels), 0.357, places=3)
    def test_gini_index(self):
        self.assertAlmostEqual(DecisionTree.gini_index(self.labels), 0.459, places=3)
unittest.main(argv=[''], verbosity=2, exit=False)