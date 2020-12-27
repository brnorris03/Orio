import pytest
import os
import sys
from os.path import abspath, dirname, join

def run_orcc(example, search="Annealing", extra_args="arg total_runs=10;"): 
    # dispatch to Orio's main
    code = join(abspath(dirname(dirname(__file__))),example)
    os.system("sed -e 's|@SEARCH@|%s|' -e 's|@EXTRA_ARGS@|%s|' %s.in > %s" % (search,extra_args,code,code))
    with pytest.raises(SystemExit) as exc:
        from orio.main.util.globals import Globals
        Globals.reset()
        import orio.main.orio_main
        cmd = ['orcc','-v','--stop-on-error','--logdir=orio/main/tuner/search/tests', code]
        print((' '.join(cmd)))
        orio.main.orio_main.start(cmd, orio.main.orio_main.C_CPP)
        captured = capsys.readouterr()
    return exc.value.code

def test_annealing(capsys, caplog):
    ret_code = run_orcc('tests/axpy4.c')
    assert ret_code == 0

