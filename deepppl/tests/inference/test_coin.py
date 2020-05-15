
from .harness import MCMCTest
from pprint import pprint

def test_coin():
    data = {}
    data['N'] = 10
    data['x'] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    
    t_coin = MCMCTest(
        name='coin',
        model_file='deepppl/tests/good/coin.stan',
        data=data
    )
    
    return t_coin.run()
    
if __name__ == "__main__":
    pprint(test_coin())