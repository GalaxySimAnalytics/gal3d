
import numpy as np

def test_fitting(gal):
    res = gal.fit(r=np.geomspace(3*gal.field.iso_pro_r[0],min(gal.field.iso_pro_r[-1],15),5))
    assert len(res)>0
    
    