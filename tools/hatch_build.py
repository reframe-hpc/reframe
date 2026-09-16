# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class ReframeBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Opt-in: only the release/packaging workflows need the docs built;
        # a plain dev install should stay fast and not require Sphinx.
        if not os.getenv('_RFM_BUILD_DOCS'):
            return

        print("Building docs...")

        # --no-install-project avoids recursing into this hook: we are
        # already building the project, so installing it here would build
        # it again.
        subprocess.run(
            ["uv", "sync", "--group", "docs", "--no-install-project"],
            check=True, cwd=self.root
        )
        subprocess.run(
            ["uv", "run", "--no-sync", "make", "-C", "docs"],
            check=True, cwd=self.root
        )
