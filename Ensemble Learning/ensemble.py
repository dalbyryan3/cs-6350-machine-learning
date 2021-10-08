class AdaBoost:
    def __init__(self) -> None:
        pass
    # Will have to make DecisionTree supports weighted training examples (Modify get_value_subset to have tuple including a weight for example? Then use weight to scale the gain value? Likely just start by putting default weight value of 1 and making sure we get same result, then add ability to give a weight for each example on training try scaling making sure nothing goes wrong (Add check around gain to make sure scale occurs...?))
    # Then implement AdaBoost
class BaggedTrees:
    def __init__(self) -> None:
        pass
class RandomForest:
    def __init__(self) -> None:
        pass