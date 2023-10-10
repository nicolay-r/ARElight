import json

from tqdm import tqdm

from arelight.run.utils import translate_value


def do_force(filename):
    with open(filename, "r") as f:
        nodes = json.load(f)

    for node in tqdm(nodes["nodes"]):
        v = node["id"]
        parts = v.split('.')
        parts[-1] = translate_value(parts[-1], src="ru", dest="en")
        node["id"] = ".".join(parts)

    with open(filename, "w") as f:
        json.dump(nodes, f)


def do_radial(filename):
    with open(filename, "r") as f:
        nodes = json.load(f)

    for n in tqdm(nodes):
        if "imports" in n:
            for node in n["imports"]:
                v = node["name"]
                parts = v.split('.')
                parts[-1] = translate_value(parts[-1], src="ru", dest="en")
                node["name"] = ".".join(parts)
        if "name" in n:
            v = n["name"]
            parts = v.split('.')
            parts[-1] = translate_value(parts[-1], src="ru", dest="en")
            n["name"] = ".".join(parts)

    with open(filename, "w") as f:
        json.dump(nodes, f)


files = [
    "./data/output-rus/radial/ru.json",
    "./data/output-rus/radial/ru_DIFFERENCE_ua.json",
    "./data/output-rus/radial/ua.json",
    "./data/output-rus/radial/ua_DIFFERENCE_ru.json",
    "./data/output-rus/radial/ua_INTERSECTION_ru.json",
]

for filepath in files:
    do_radial(filepath)
