import sys
import pytest
import glob
import logging
from os.path import abspath, dirname, join, normpath


def run_orcc(example): 
    # dispatch to Orio's main
    code_dir = join(abspath(dirname(dirname(__file__))),'..','examples')
    code = normpath(join(code_dir,example))
    spec = normpath(join(code_dir,'axpy5.spec'))
    with pytest.raises(SystemExit) as exc:
        from orio.main.util.globals import Globals
        Globals.reset()
        import orio.main.orio_main
        cmd = ['orcc','-v','--stop-on-error','--logdir=orio/tests', '-s', spec, code]
        print((' '.join(cmd)))
        orio.main.orio_main.start(cmd, orio.main.orio_main.C_CPP)
        captured = capsys.readouterr()
        #logging.getLogger().info(captured)
    return exc.value.code

def test_examples_axpy5_nospec(capsys, caplog):
    ret_code = run_orcc('axpy5-nospec.c')
    assert ret_code == 0

