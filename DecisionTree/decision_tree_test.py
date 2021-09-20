from logging import PlaceHolder
import unittest
from decision_tree import DecisionTree

class DecisionTreeTester(unittest.TestCase):
    def setUp(self):
        self.labels = [False,False,True,True,True,False,True,False,True,True,True,True,True,False]
        self.S = [
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
            self.be('Overcast','Hot', 'Normal', 'Weak'),
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
    def test_get_attribute_values(self):
        self.assertSetEqual(DecisionTree.get_attribute_values(self.S, 'Outlook'), {'Sunny','Overcast','Rainy'})
        self.assertSetEqual(DecisionTree.get_attribute_values(self.S, 'Temperature'), {'Hot','Medium','Cold'})
        self.assertSetEqual(DecisionTree.get_attribute_values(self.S, 'Humidity'), {'High','Normal'})
        self.assertSetEqual(DecisionTree.get_attribute_values(self.S, 'Wind'), {'Strong','Weak'})
    def test_get_value_subset(self):
        self.assertSetEqual(set(DecisionTree.get_value_subset(self.S, self.labels, 'Outlook').keys()), {'Sunny','Overcast','Rainy'})
        self.assertSetEqual(set(DecisionTree.get_value_subset(self.S, self.labels, 'Temperature').keys()), {'Hot','Medium','Cold'})
        self.assertSetEqual(set(DecisionTree.get_value_subset(self.S, self.labels, 'Humidity').keys()), {'High','Normal'})
        self.assertSetEqual(set(DecisionTree.get_value_subset(self.S, self.labels, 'Wind').keys()), {'Strong','Weak'})

        S_outlook = DecisionTree.get_value_subset(self.S, self.labels, 'Outlook')
        sunny_label_vals = []
        S_sunny = S_outlook['Sunny']
        overcast_label_vals = []
        S_overcast = S_outlook['Overcast']
        rainy_label_vals = []
        S_rainy = S_outlook['Rainy']
        for label in S_sunny:
            sunny_label_vals.append(label[1])
        self.assertListEqual(sunny_label_vals, [False, False, False, True, True])
        for label in S_overcast:
            overcast_label_vals.append(label[1])
        self.assertListEqual(overcast_label_vals, [True, True, True, True])
        for label in S_rainy:
            rainy_label_vals.append(label[1])
        self.assertListEqual(rainy_label_vals, [True, True, False, True, False])

    def test_gain(self):
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Outlook', metric=DecisionTree.entropy)[0], 0.171, places=3)
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Temperature', metric=DecisionTree.entropy)[0], 0.020, places=3)
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Humidity', metric=DecisionTree.entropy)[0], 0.105, places=3)
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Wind', metric=DecisionTree.entropy)[0], 0.0334, places=3)

        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Outlook', metric=DecisionTree.majority_error)[0], 0.0714, places=3)
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Temperature', metric=DecisionTree.majority_error)[0], 0.00, places=3)
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Humidity', metric=DecisionTree.majority_error)[0], 0.0714, places=3)
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Wind', metric=DecisionTree.majority_error)[0], 0.00, places=3)

        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Outlook', metric=DecisionTree.gini_index)[0], 0.116, places=3)
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Temperature', metric=DecisionTree.gini_index)[0], 0.019, places=3)
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Humidity', metric=DecisionTree.gini_index)[0], 0.092, places=3)
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Wind', metric=DecisionTree.gini_index)[0], 0.031, places=3)


unittest.main(argv=[''], verbosity=2, exit=False)