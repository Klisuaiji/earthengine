# -*- coding: utf-8 -*-
"""Run the upstream worldengine CLI from its own repo copy."""
import sys, os
UP = r"D:\Qq203\Downloads\earthengine-master\_reference_repos\worldengine"
sys.path.insert(0, UP)
os.chdir(UP)
sys.argv = [
    "worldengine",
    "--seed", "1234567",
    "--width", "512", "--height", "256",
    "--plates", "6",
    "--step", "warm",
    "-q", "--outputdir", "upstream_out",
]
from worldengine.cli.main import main
main()
