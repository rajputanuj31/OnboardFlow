from __future__ import annotations

import io

from unidiff.patch import PatchSet

from onboardflow_evals.data_prep import enclosing_functions, gold_from_patch, touched_source_lines

SOURCE = """\
import os


class Session:
    def __init__(self):
        self.headers = {}

    def request(self, method, url):
        prep = self.prepare(method, url)
        return self.send(prep)


def helper():
    return 1
"""

PATCH = """\
diff --git a/pkg/sessions.py b/pkg/sessions.py
--- a/pkg/sessions.py
+++ b/pkg/sessions.py
@@ -8,4 +8,5 @@ class Session:
     def request(self, method, url):
         prep = self.prepare(method, url)
+        prep.check()
         return self.send(prep)

@@ -13,2 +14,2 @@ def helper():
 def helper():
-    return 1
+    return 2
"""


def test_touched_lines_map_to_original_file() -> None:
    patched = next(iter(PatchSet(io.StringIO(PATCH))))
    # The added line is anchored to line 9 (the line above it); the removed line is 14.
    assert touched_source_lines(patched) == {9, 14}


def test_enclosing_functions_are_qualified() -> None:
    assert enclosing_functions(SOURCE, [9, 14]) == ["Session.request", "helper"]
    assert enclosing_functions(SOURCE, [1]) == []  # module level


def test_gold_from_patch_lists_files_and_functions() -> None:
    files, functions = gold_from_patch(PATCH, {"pkg/sessions.py": SOURCE})
    assert files == ["pkg/sessions.py"]
    assert functions == ["pkg/sessions.py::Session.request", "pkg/sessions.py::helper"]
