import unittest
import MyFunctions

class MyTestCase(unittest.TestCase):
    def test_something(self):
        # Let's load a known file, parse it, and then validate it is as expected
        trainX, testX, trainY, testY, trainYRnd, testYRnd = MyFunctions.parse_data_to_trainXY_plusRandY ("DAX_Prices_WithMLPercentages.csv", "BuyWeightingRule")

        print("TrainX:")
        print(trainX)
        print("TestX")
        print(testX)
        print("TrainY:")
        print(trainY)
        print("TestY:")
        print(testY)
        print("trainXRnd")
        print(trainYRnd)
        print("testYRnd")
        print(testYRnd)

        print("Shapes")
        print(trainX.shape)
        print(testX.shape)
        print(trainY.shape)
        print(testY.shape)
        print(trainYRnd.shape)
        print(testYRnd.shape)

        self.assertEqual(len(trainY), len(trainYRnd), "TestX Length")
        self.assertEqual(len(testY), len(testYRnd), "TestX Length")

#        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
