from app.postprocess import enforce_plate_mask,is_valid_plate

def test_valid():
    assert is_valid_plate('A123BC77')
    assert is_valid_plate('M555OP716')

def test_mask():
    assert enforce_plate_mask('A123BC77')=='A123BC77'
