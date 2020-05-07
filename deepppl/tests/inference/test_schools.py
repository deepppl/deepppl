from .harness import MCMCTest
from pprint import pprint

def test_schools():
    data = {}
    data['N'] = 8
    data['y'] = [28, 8, -3, 7, -1, 1, 18, 12]
    data['sigma_y'] = [15, 10, 16, 11, 9, 11, 10, 18]
        
    test_schools = MCMCTest(
        name='schools',
        model_file='deepppl/tests/good/schools.stan',
        data=data
    )
    
    return test_schools.run()
    
if __name__ == "__main__":
    pprint(test_schools())