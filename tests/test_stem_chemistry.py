import pytest

from vifasttts.frontend.normalizer import integer_to_words, normalize_text
from vifasttts.frontend.stem import read_chemical_formula

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

@pytest.mark.parametrize("source,expected",[
 ("O2","oxi hai"),("O3","oxi ba"),("H2","hiđrô hai"),("Cl2","clo hai"),
 ("Fe3+","sắt ba dương"),("O2-","oxi hai âm"),("NH4+","nitơ hiđrô bốn dương"),
])
def test_simple_molecules_and_ions(source,expected):assert normalize_text(source)==expected

@pytest.mark.parametrize("code",["MP3","N95","K2","RTX3080","v2","B2B"])
def test_product_codes_not_chemistry(code):assert normalize_text(code)==code
