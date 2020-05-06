from .harness import MCMCTest
from pprint import pprint
import pytest

@pytest.mark.xfail(reason='Numpyro cannot compile boolean condition')
def test_multi_modal():
    data = {}
        
    t_multi_modal = MCMCTest(
        name='multi_modal',
        model_file='deepppl/tests/good/paper/multimodal.stan',
        data=data
    )
    
    return t_multi_modal.run()
    
if __name__ == "__main__":
    pprint(test_multi_modal())