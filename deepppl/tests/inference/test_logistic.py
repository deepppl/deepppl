from .harness import MCMCTest
from pprint import pprint

@pytest.mark.xfail('Numpyro cannot find valid initial parameters')
def test_logistic():
    data = {}
    data['N'] = 7
    data['M'] = 2
    data['y'] = [0, 1, 1, 1, 1, 1, 1]
    data['x'] = [[1, 1.329799263],
                 [1, 1.272429321],
                 [1, -1.539950042],
                 [1, -0.928567035],
                 [1, -0.294720447],
                 [1, -0.005767173],
                 [1, 2.404653389]]

    t_logistic = MCMCTest(
        name='logistic',
        model_file='deepppl/tests/good/paper/logistic.stan',
        data=data
    )
    
    return t_logistic.run_pyro()

if __name__ == "__main__":
    pprint(test_logistic())