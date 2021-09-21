import unittest
from decision_tree import DecisionTree, Node

class DecisionTreeTester(unittest.TestCase):
    # Helper and setup methods
    def setUp(self):
        self.attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
        self.attribute_possible_vals = {
            self.attributes[0] : ['Sunny', 'Overcast', 'Rainy'],
            self.attributes[1] : ['Hot', 'Medium', 'Cold'],
            self.attributes[2] : ['High', 'Normal', 'Low'],
            self.attributes[3] : ['Strong', 'Weak']
        }
        self.labels = [False,False,True,True,True,False,True,False,True,True,True,True,True,False]
        self.S = [
            self.build_example('Sunny','Hot', 'High', 'Weak'),
            self.build_example('Sunny','Hot', 'High', 'Strong'),
            self.build_example('Overcast','Hot', 'High', 'Weak'),
            self.build_example('Rainy','Medium', 'High', 'Weak'),
            self.build_example('Rainy','Cold', 'Normal', 'Weak'),
            self.build_example('Rainy','Cold', 'Normal', 'Strong'),
            self.build_example('Overcast','Cold', 'Normal', 'Strong'),
            self.build_example('Sunny','Medium', 'High', 'Weak'),
            self.build_example('Sunny','Cold', 'Normal', 'Weak'),
            self.build_example('Rainy','Medium', 'Normal', 'Weak'),
            self.build_example('Sunny','Medium', 'Normal', 'Strong'),
            self.build_example('Overcast','Medium', 'High', 'Strong'),
            self.build_example('Overcast','Hot', 'Normal', 'Weak'),
            self.build_example('Rainy','Medium', 'High', 'Strong')
        ]
    def build_example(self, outlook, temperature, humidity, wind):
        # Build example
        return {self.attributes[0]:outlook, self.attributes[1]:temperature,self.attributes[2]:humidity, self.attributes[3]:wind}

        
    # Testing methods
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
        sunny_label_vals = S_outlook['Sunny'][1]
        self.assertListEqual(sunny_label_vals, [False, False, False, True, True])
        overcast_label_vals = S_outlook['Overcast'][1]
        self.assertListEqual(overcast_label_vals, [True, True, True, True])
        rainy_label_vals = S_outlook['Rainy'][1]
        self.assertListEqual(rainy_label_vals, [True, True, False, True, False])

    def test_gain(self):
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Outlook')[0], 0.171, places=3)
        self.assertAlmostEqual(DecisionTree.gain(self.S, self.labels, 'Temperature')[0], 0.020, places=3)
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

    def test_find_best_attribute(self):
        (best_attr, Sv_dict, best_gain) = DecisionTree.find_best_attribute(self.S, self.labels, self.attributes)
        self.assertEqual(best_attr, 'Outlook')
        self.assertAlmostEqual(best_gain, 0.171, places=3)
        sunny_label_vals = Sv_dict['Sunny'][1]
        self.assertListEqual(sunny_label_vals, [False, False, False, True, True])
        overcast_label_vals = Sv_dict['Overcast'][1]
        self.assertListEqual(overcast_label_vals, [True, True, True, True])
        rainy_label_vals = Sv_dict['Rainy'][1]
        self.assertListEqual(rainy_label_vals, [True, True, False, True, False])

        (best_attr, Sv_dict, best_gain) = DecisionTree.find_best_attribute(self.S, self.labels, self.attributes, metric=DecisionTree.majority_error)
        self.assertTrue(best_attr == 'Outlook' or best_attr =='Humidity')
        self.assertAlmostEqual(best_gain, 0.0714, places=3)

        (best_attr, Sv_dict, best_gain) = DecisionTree.find_best_attribute(self.S, self.labels, self.attributes, metric=DecisionTree.gini_index)
        self.assertEqual(best_attr, 'Outlook')
        self.assertAlmostEqual(best_gain, 0.116, places=3)
        sunny_label_vals = Sv_dict['Sunny'][1]
        self.assertListEqual(sunny_label_vals, [False, False, False, True, True])
        overcast_label_vals = Sv_dict['Overcast'][1]
        self.assertListEqual(overcast_label_vals, [True, True, True, True])
        rainy_label_vals = Sv_dict['Rainy'][1]
        self.assertListEqual(rainy_label_vals, [True, True, False, True, False])
    
    def test_ID3_train(self):
        dc = DecisionTree(self.attribute_possible_vals)

        dc.train(self.S, self.attributes, self.labels)
        tree_str = dc.visualize_tree()
        self.assertEqual(tree_str[0], '---->Outlook    ')
        self.assertEqual(tree_str[1], 'Outlook--Sunny-->Humidity    Outlook--Overcast-->True    Outlook--Rainy-->Wind    ')
        self.assertEqual(tree_str[2], 'Humidity--High-->False    Humidity--Normal-->True    Humidity--Low-->False    Wind--Strong-->False    Wind--Weak-->True    ')

        dc.train(self.S, self.attributes, self.labels, max_depth=1)
        tree_str = dc.visualize_tree()
        self.assertEqual(tree_str[0], '---->Outlook    ')
        self.assertEqual(tree_str[1], 'Outlook--Sunny-->False    Outlook--Overcast-->True    Outlook--Rainy-->True    ')

unittest.main(argv=[''], verbosity=2, exit=False)