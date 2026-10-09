from __future__ import annotations

import re
from collections.abc import Callable

NumberReader = Callable[[int], str]

GREEK = {
    "alpha":"an pha", "beta":"bê ta", "gamma":"gam ma", "delta":"đen ta",
    "epsilon":"ép xi lon", "theta":"thê ta", "lambda":"lam đa", "mu":"miu",
    "pi":"pi", "rho":"rô", "sigma":"xích ma", "phi":"phi", "omega":"ô mê ga",
}

MATH_COMMANDS = {
    "times":"nhân", "cdot":"nhân", "div":"chia", "pm":"cộng hoặc trừ",
    "neq":"khác", "ne":"khác", "le":"nhỏ hơn hoặc bằng", "leq":"nhỏ hơn hoặc bằng",
    "ge":"lớn hơn hoặc bằng", "geq":"lớn hơn hoặc bằng", "approx":"xấp xỉ",
    "infty":"vô cùng", "sum":"tổng", "prod":"tích", "int":"tích phân",
    "partial":"đạo hàm riêng", "nabla":"na bla", "rightarrow":"suy ra",
    "Rightarrow":"suy ra", "leftarrow":"suy ra ngược", "in":"thuộc",
    "notin":"không thuộc", "subset":"là tập con của", "subseteq":"là tập con hoặc bằng của",
    "cup":"hợp", "cap":"giao", "forall":"với mọi", "exists":"tồn tại",
}

VARIABLES = {
    "x":"ích", "y":"i dài", "z":"dét", "a":"a", "b":"bê", "c":"xê",
    "d":"đê", "e":"e", "f":"ép", "g":"giê", "h":"hát", "i":"i",
    "j":"giây", "k":"ca", "m":"em", "n":"en", "p":"pê", "q":"quy",
    "r":"e rờ", "s":"ét", "t":"tê", "u":"u", "v":"vê", "w":"vê kép",
}

ELEMENTS = {
    "H":"hiđrô", "He":"heli", "Li":"liti", "Be":"beri", "B":"bo", "C":"cacbon",
    "N":"nitơ", "O":"oxi", "F":"flo", "Ne":"neon", "Na":"natri", "Mg":"magiê",
    "Al":"nhôm", "Si":"silic", "P":"phốt pho", "S":"lưu huỳnh", "Cl":"clo",
    "K":"kali", "Ca":"canxi", "Fe":"sắt", "Cu":"đồng", "Zn":"kẽm", "Ag":"bạc",
    "Au":"vàng", "Hg":"thủy ngân", "Pb":"chì", "Sn":"thiếc", "Mn":"mangan",
    "Cr":"crom", "Co":"coban", "Ni":"niken", "Br":"brom", "I":"iốt", "Ba":"bari",
}

UNITS = {
    "m":"mét", "km":"ki lô mét", "cm":"xen ti mét", "mm":"mi li mét",
    "s":"giây", "ms":"mi li giây", "h":"giờ", "Hz":"héc", "kHz":"ki lô héc",
    "MHz":"mê ga héc", "GHz":"gi ga héc", "kg":"ki lô gam", "g":"gam",
    "N":"niu tơn", "Pa":"pát can", "kPa":"ki lô pát can", "J":"jun", "W":"oát",
    "kW":"ki lô oát", "V":"vôn", "A":"ampe", "C":"cu lông", "Ω":"ôm",
    "K":"ken vin", "°C":"độ xê", "mol":"mol", "L":"lít", "mL":"mi li lít",
}


def _read_int(text: str, read_number: NumberReader) -> str:
    try: return read_number(int(text))
    except ValueError: return text


def _balanced_argument(text: str, command: str) -> tuple[str, str] | None:
    prefix = "\\" + command + "{"
    if not text.startswith(prefix): return None
    depth = 1
    for i in range(len(prefix), len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return text[len(prefix):i], text[i+1:]
    return None


def read_chemical_formula(formula: str, read_number: NumberReader) -> str:
    formula = formula.strip().replace("_{", "").replace("}", "")
    parts=[]; pos=0
    token_re=re.compile(r"([A-Z][a-z]?)(\d*)|([()])|(\d+)|([+\-])")
    for m in token_re.finditer(formula):
        if m.start()!=pos: return formula
        pos=m.end()
        if m.group(1):
            element=m.group(1); parts.append(ELEMENTS.get(element, element))
            if m.group(2): parts.append(_read_int(m.group(2),read_number))
        elif m.group(3): parts.append("mở ngoặc" if m.group(3)=="(" else "đóng ngoặc")
        elif m.group(4): parts.append(_read_int(m.group(4),read_number))
        else: parts.append("dương" if m.group(5)=="+" else "âm")
    return " ".join(parts) if pos==len(formula) and parts else formula


def read_unit_expression(expr: str, read_number: NumberReader) -> str:
    expr=expr.strip()
    expr=re.sub(r"([A-Za-z°Ω]+)\^?([23])\b", lambda m: f"{UNITS.get(m.group(1),m.group(1))} {'bình phương' if m.group(2)=='2' else 'lập phương'}", expr)
    pieces=re.split(r"([/·*])",expr); out=[]
    for p in pieces:
        p=p.strip()
        if not p: continue
        if p=="/": out.append("trên")
        elif p in {"·","*"}: out.append("nhân")
        else: out.append(UNITS.get(p,p))
    return " ".join(out)


def read_math_expression(expr: str, read_number: NumberReader) -> str:
    expr=expr.strip().replace("\\\\", "\\")
    expr=re.sub(r"\\+(?=[A-Za-z])", r"\\", expr)
    # Structured LaTeX commands, recursively parsed.
    while "\\frac{" in expr:
        m=re.search(r"\\frac\{([^{}]+)\}\{([^{}]+)\}",expr)
        if not m: break
        repl=f"phân số {read_math_expression(m.group(1),read_number)} trên {read_math_expression(m.group(2),read_number)}"
        expr=expr[:m.start()]+repl+expr[m.end():]
    expr=re.sub(r"\\sqrt\[([^]]+)\]\{([^{}]+)\}",lambda m:f"căn bậc {read_math_expression(m.group(1),read_number)} của {read_math_expression(m.group(2),read_number)}",expr)
    expr=re.sub(r"\\sqrt\{([^{}]+)\}",lambda m:f"căn bậc hai của {read_math_expression(m.group(1),read_number)}",expr)
    expr=re.sub(r"\\text\{([^{}]*)\}",r"\1",expr)
    expr=re.sub(r"\\(?:mathrm|mathbf|mathit)\{([^{}]*)\}",r"\1",expr)
    for k,v in sorted({**GREEK,**MATH_COMMANDS}.items(), key=lambda item: len(item[0]), reverse=True):
        expr=expr.replace("\\" + k, v)
    expr=expr.replace("\\,"," ").replace("\\;"," ").replace("\\!","")
    # Superscript/subscript before generic punctuation.
    expr=re.sub(r"\^\{([^{}]+)\}",lambda m:" mũ "+read_math_expression(m.group(1),read_number),expr)
    expr=re.sub(r"\^([0-9A-Za-z])",lambda m:" mũ "+read_math_expression(m.group(1),read_number),expr)
    expr=re.sub(r"_\{([^{}]+)\}",lambda m:" chỉ số "+read_math_expression(m.group(1),read_number),expr)
    expr=re.sub(r"_([0-9A-Za-z])",lambda m:" chỉ số "+read_math_expression(m.group(1),read_number),expr)
    replacements=[("<="," nhỏ hơn hoặc bằng "),(">="," lớn hơn hoặc bằng "),("!="," khác "),("=>"," suy ra "),("->"," suy ra "),("±"," cộng hoặc trừ "),("×"," nhân "),("÷"," chia "),("="," bằng "),("+"," cộng "),("-"," trừ "),("<"," nhỏ hơn "),(">"," lớn hơn "),("/"," trên "),("·"," nhân "),("*"," nhân "),("%"," phần trăm ")]
    for a,b in replacements: expr=expr.replace(a,b)
    expr=re.sub(r"\b\d+\b",lambda m:read_number(int(m.group())),expr)
    expr=re.sub(r"\b([A-Za-z])\b",lambda m:VARIABLES.get(m.group().lower(),m.group()),expr)
    expr=expr.replace("("," mở ngoặc ").replace(")"," đóng ngoặc ").replace("["," mở ngoặc vuông ").replace("]"," đóng ngoặc vuông ").replace("{"," mở ngoặc nhọn ").replace("}"," đóng ngoặc nhọn ")
    return re.sub(r"\s+"," ",expr).strip()


def normalize_stem(text: str, read_number: NumberReader) -> str:
    """Normalize explicit LaTeX, chemistry formulas, common equations and SI units."""
    # Accept text originating from JSON/Python strings with doubled backslashes.
    text=text.replace("\\\\", "\\")
    # Explicit chemistry commands are handled before generic LaTeX.
    text=re.sub(r"\\(?:ce|chem)\{([^{}]+)\}",lambda m:read_chemical_formula(m.group(1),read_number),text)
    # Display and inline LaTeX.
    patterns=[r"\$\$([^$]+)\$\$",r"\$([^$]+)\$",r"\\\((.*?)\\\)",r"\\\[(.*?)\\\]"]
    for pat in patterns: text=re.sub(pat,lambda m:read_math_expression(m.group(1),read_number),text,flags=re.S)
    # Safe standalone LaTeX constructs that have explicit brace boundaries.
    text=re.sub(r"\\frac\{[^{}]+\}\{[^{}]+\}", lambda m: read_math_expression(m.group(), read_number), text)
    text=re.sub(r"\\sqrt(?:\[[^]]+\])?\{[^{}]+\}", lambda m: read_math_expression(m.group(), read_number), text)
    # Common standalone chemistry formulas containing digits or multiple element symbols.
    chem_pat=re.compile(r"(?<!\w)(?=[A-Z][A-Za-z0-9()]*\d)([A-Z][A-Za-z0-9()]*)(?!\w)")
    text=chem_pat.sub(lambda m:read_chemical_formula(m.group(1),read_number),text)
    # Number followed by recognized unit expression.
    unit_keys="|".join(sorted((re.escape(x) for x in UNITS),key=len,reverse=True))
    text=re.sub(rf"(?<!\w)(\d+(?:[,.]\d+)?)\s*({unit_keys})(?:\^?([23]))?(?:/({unit_keys})(?:\^?([23]))?)?",lambda m:_read_measure(m,read_number),text)
    # Composite units may follow a LaTeX expression whose number has already been verbalized.
    text=re.sub(rf"(?<!\w)({unit_keys})(?:\^?([23]))?/({unit_keys})(?:\^?([23]))?(?!\w)", lambda m: _read_composite_unit(m), text)
    return re.sub(r"\s+"," ",text).strip()


def _read_measure(m: re.Match[str], read_number: NumberReader) -> str:
    number=m.group(1).replace(",",".")
    if "." in number:
        whole,frac=number.split(".",1); n=read_number(int(whole))+" phẩy "+" ".join(read_number(int(x)) for x in frac)
    else: n=read_number(int(number))
    unit=UNITS[m.group(2)]
    if m.group(3): unit+= " bình phương" if m.group(3)=="2" else " lập phương"
    if m.group(4):
        denominator=UNITS[m.group(4)]
        if m.group(5): denominator+= " bình phương" if m.group(5)=="2" else " lập phương"
        unit+= " trên "+denominator
    return n+" "+unit


def _read_composite_unit(m: re.Match[str]) -> str:
    numerator=UNITS[m.group(1)]
    if m.group(2): numerator += " bình phương" if m.group(2)=="2" else " lập phương"
    denominator=UNITS[m.group(3)]
    if m.group(4): denominator += " bình phương" if m.group(4)=="2" else " lập phương"
    return numerator + " trên " + denominator
