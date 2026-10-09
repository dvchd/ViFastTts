import pytest
from vifasttts.frontend.stem import read_chemical_formula
from vifasttts.frontend.normalizer import integer_to_words, normalize_text
R=lambda n:integer_to_words(n,"north")
@pytest.mark.parametrize("source,expected",[
 ("H2O","hiđrô hai oxi"),("CO2","cacbon oxi hai"),("NaCl","natri clo"),
 ("H2SO4","hiđrô hai lưu huỳnh oxi bốn"),("CaCO3","canxi cacbon oxi ba"),
 ("Fe2O3","sắt hai oxi ba"),("C6H12O6","cacbon sáu hiđrô mười hai oxi sáu"),
 ("Ca(OH)2","canxi mở ngoặc oxi hiđrô đóng ngoặc hai"),
])
def test_chem(source,expected):assert read_chemical_formula(source,R)==expected

def test_explicit_chem_command():assert normalize_text(r"\\ce{H2O}")=="hiđrô hai oxi"
def test_automatic_formula():assert normalize_text("CO2 và H2O")=="cacbon oxi hai và hiđrô hai oxi"
def test_vi_not_captured_as_chemistry():assert normalize_text("VI vi")=="sáu vi"
