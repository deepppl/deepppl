from .harness import MCMCTest
from pprint import pprint

def test_double_gaussian():
    t_double_gaussian = MCMCTest(
        name='double_gaussian',
        model_file='deepppl/tests/good/double_gaussian.stan'
    )
    
    return t_double_gaussian.run()
    
if __name__ == "__main__":
    pprint(test_double_gaussian())