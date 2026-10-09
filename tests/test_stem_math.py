import pytest
from vifasttts.frontend.stem import read_math_expression
from vifasttts.frontend.normalizer import integer_to_words, normalize_text
R=lambda n:integer_to_words(n,"north")
@pytest.mark.parametrize("source,expected",[
 (r"x^2 + y^2 = z^2","ích mũ hai cộng i dài mũ hai bằng dét mũ hai"),
 (r"\\frac{1}{2}","phân số một trên hai"),
 (r"\\sqrt{x}","căn bậc hai của ích"),
 (r"\\sqrt[3]{8}","căn bậc ba của tám"),
 (r"a \\leq b","a nhỏ hơn hoặc bằng bê"),
 (r"\\sum_{i=1}^{n} i","tổng chỉ số i bằng một mũ en i"),
 (r"\\alpha + \\beta","an pha cộng bê ta"),
 (r"(a+b)/c","mở ngoặc a cộng bê đóng ngoặc trên xê"),
])
def test_read_math(source,expected):assert read_math_expression(source,R)==expected

def test_inline_latex():
 assert normalize_text(r"Ta có $x^2 + 1 = 5$.")=="Ta có ích mũ hai cộng một bằng năm."

def test_latex_fraction():
 assert normalize_text(r"Giá trị là \\(\\frac{3}{4}\\).")=="Giá trị là phân số ba trên bốn."


def test_standalone_safe_latex():
 assert normalize_text(r"\frac{3}{4}")=="phân số ba trên bốn"
 assert normalize_text(r"\sqrt{9}")=="căn bậc hai của chín"
