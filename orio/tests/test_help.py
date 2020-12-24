import pytest
import sys

def test_orcc_help(capsys):
    # dispatch to Orio's main
    with pytest.raises(SystemExit) as exc:
        import orio.main.orio_main
        orio.main.orio_main.start(['orcc','--help'], orio.main.orio_main.C_CPP)
        captured = capsys.readouterr()
    assert exc.value.code == 0
