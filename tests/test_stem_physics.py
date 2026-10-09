import pytest
from vifasttts.frontend.normalizer import normalize_text
@pytest.mark.parametrize("source,expected",[
 ("10 m","mười mét"),("20 km","hai mươi ki lô mét"),("9,8 m/s2","chín phẩy tám mét trên giây bình phương"),
 ("50 Hz","năm mươi héc"),("220 V","hai trăm hai mươi vôn"),("5 A","năm ampe"),
 ("100 W","một trăm oát"),("25 °C","hai mươi lăm độ xê"),("2 kg","hai ki lô gam"),
 ("3 m2","ba mét bình phương"),("4 m3","bốn mét lập phương"),
])
def test_units(source,expected):assert normalize_text(source)==expected

def test_formula_and_unit():
 assert normalize_text(r"$v = 10$ m/s")=="vê bằng mười mét trên giây"
