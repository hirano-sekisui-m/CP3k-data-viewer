import subprocess
import os

# We need to make sure the pairs loop duplicate fix logic gets pushed,
# I used replace_with_git_merge_diff earlier for plot and the others using scripts... oh wait,
# the git diff already shows the fix in the notebook:
# -            pairs = [(b, c) for b in bases for c in value_cols if c != b]
# +            pairs = []
# +            for b in bases:
# +                for c in value_cols:
# +                    if c != b and (b, c) not in pairs and (c, b) not in pairs:
# +                        pairs.append((b, c))

print("Ready to commit")
