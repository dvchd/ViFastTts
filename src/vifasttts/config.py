import yaml
def load_config(p):
 return yaml.safe_load(open(p,encoding="utf-8"))
