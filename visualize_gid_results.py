#!/usr/bin/env python3
"""
Visualize GiD .post.res results on GiD .post.msh meshes, with profile focus.

Features:
- Parse GiD mesh (.post.msh): nodes, triangle/quad elements, line meshes
- Parse GiD results (.post.res): Scalar/Vector OnNodes blocks
- List available result names/steps
- Plot domain map + profile for a selected result

Examples:
  python Eemdijk/tools/visualize_gid_results.py       Eemdijk/20_Bovenbelasting       --list

  python Eemdijk/tools/visualize_gid_results.py       Eemdijk/20_Bovenbelasting       --result WATER_PRESSURE --step latest

  python Eemdijk/tools/visualize_gid_results.py       Eemdijk/20_Bovenbelasting       --result REACTION --component magnitude --step latest
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

from kratos_visualization_tools._compat import install_facade as _install_facade
from kratos_visualization_tools import gid as _implementation

_install_facade(_sys.modules[__name__], _implementation._compat_owners, _implementation._compat_modules)

if __name__ == "__main__":
    main()
