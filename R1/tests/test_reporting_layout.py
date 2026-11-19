"""Reporting works with a manuscript checkout and in a standalone release."""
import json

from reporting import manuscript_sections


def test_standalone_sections_preserve_multiple_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(manuscript_sections, 'PAPER', tmp_path / 'absent_paper')
    protocols = tmp_path / 'results' / 'models' / 'protocols'
    monkeypatch.setattr(manuscript_sections, 'PROTOCOLS', protocols)
    manuscript_sections.write_section('results', 'First result\n')
    manuscript_sections.write_section('conclusion', 'Conclusion\n')
    manuscript_sections.write_section('results', 'Updated result\n')
    assert json.loads((protocols / 'manuscript_sections.json').read_text()) == {
        'results': 'Updated result\n', 'conclusion': 'Conclusion\n'}
    assert not (tmp_path / 'absent_paper').exists()


def test_manuscript_sync_replaces_only_named_section(tmp_path, monkeypatch):
    monkeypatch.setattr(manuscript_sections, 'PAPER', tmp_path)
    path = tmp_path / 'paper.tex'
    path.write_text('Before\n% BEGIN FEDPERG SECTION results\nOld\n'
                    '% END FEDPERG SECTION results\nAfter\n', encoding='utf-8')
    manuscript_sections.write_section('results', 'New')
    assert path.read_text(encoding='utf-8') == (
        'Before\n% BEGIN FEDPERG SECTION results\nNew\n'
        '% END FEDPERG SECTION results\nAfter\n')
