"""Update generated sections inside the single clean manuscript source."""
import json
import re

from fedperg.paths import ROOT, PROTOCOLS

PAPER = ROOT.parents[1] / 'paper' / 'INS' / 'R1'


def write_section(name, content):
    path = PAPER / 'paper.tex'
    if not path.is_file():
        PROTOCOLS.mkdir(parents=True, exist_ok=True)
        output = PROTOCOLS / 'manuscript_sections.json'
        sections = json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
        sections[name] = content
        output.write_text(json.dumps(sections, indent=2), encoding='utf-8')
        return
    source = path.read_text(encoding='utf-8-sig')
    begin = '% BEGIN FEDPERG SECTION ' + name
    end = '% END FEDPERG SECTION ' + name
    pattern = re.compile(r'^' + re.escape(begin) + r'\n.*?^' + re.escape(end) + r'\n', re.M | re.S)
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one inline manuscript section {name!r}; found {len(matches)}')
    body = content if content.endswith('\n') else content + '\n'
    replacement = begin + '\n' + body + end + '\n'
    revised = pattern.sub(lambda _: replacement, source, count=1)
    if revised != source:
        path.write_text(revised, encoding='utf-8')
