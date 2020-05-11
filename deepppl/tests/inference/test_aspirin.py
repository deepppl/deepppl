from .harness import MCMCTest
import numpy as np
from pprint import pprint

# Warning: Generated quantities does not work with numpyro on this example
# (In place mutation of array)

def test_aspirin():
    data = {}
    data['y'] = [2.77, 2.50, 1.84, 2.56, 2.31, -1.15]
    data['s'] = [1.65, 1.31, 2.34, 1.67, 1.98, 0.90]
    data['N'] = len(data['y'])
    data['mu_loc'] = np.mean(data['y'])
    data['mu_scale'] = 5 * np.std(data['y'])
    data['tau_scale'] = 2.5 * np.std(data['y'])
    data['tau_df'] = 4
     
    t_aspirin = MCMCTest(
        name='aspirin',
        model_file='deepppl/tests/good/aspirin.stan',
        data=data
    )

    return t_aspirin.run()
    
if __name__ == "__main__":
    pprint(test_aspirin())