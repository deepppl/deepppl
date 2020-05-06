from .harness import MCMCTest
from pprint import pprint

def test_neal_funnel():
    data = {}
        
    test_neal_funnel = MCMCTest(
        name='neal_funnel',
        model_file='deepppl/tests/good/paper/neal_funnel.stan',
        data=data
    )
    
    return test_neal_funnel.run()
    
if __name__ == "__main__":
    pprint(test_neal_funnel())