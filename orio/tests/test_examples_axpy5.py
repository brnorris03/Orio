import sys
import pytest
import glob
import logging
from os.path import abspath, dirname, join


def run_orcc(example): 
    # dispatch to Orio's main
    code = join(abspath(dirname(dirname(__file__))),'..','examples',example)
    with pytest.raises(SystemExit) as exc:
        from orio.main.util.globals import Globals
        Globals.reset()
        import orio.main.orio_main
        cmd = ['orcc','-v','--stop-on-error','--logdir=orio/tests', code]
        print((' '.join(cmd)))
        orio.main.orio_main.start(cmd, orio.main.orio_main.C_CPP)
        captured = capsys.readouterr()
        #logging.getLogger().info(captured)
    return exc.value.code

def test_examples_axpy5(capsys, caplog):
    ret_code = run_orcc('axpy5.c')
    assert ret_code == 0

