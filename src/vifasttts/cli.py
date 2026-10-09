import json,typer
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.syllable import parse_syllable
app=typer.Typer()
@app.command("parse-syllable")
def one(value:str): print(json.dumps(parse_syllable(value).to_dict(),ensure_ascii=False,default=str,indent=2))
@app.command("parse")
def parse(value:str): print(json.dumps([x.to_dict() for x in parse_text(value)],ensure_ascii=False,default=str,indent=2))
