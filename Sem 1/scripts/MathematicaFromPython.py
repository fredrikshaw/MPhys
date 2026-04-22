import subprocess
import shlex
import math
import sys

# path to your WL script
wl_script = "C:\\Users\\fredr\\Documents\\Uni\\Year 4\\MPhys\\MPhys\\Mathematica\\RunFromPythonTest.wl"

# command to run (use wolframscript provided with Mathematica / WolframEngine)
cmd = f'wolframscript -file "{wl_script}"'

# run wolframscript and capture stdout/stderr
proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)

# debug output
print("=== Mathematica stdout ===")
print(proc.stdout)
print("=== Mathematica stderr ===")
print(proc.stderr)

if proc.returncode != 0:
    raise RuntimeError(f"wolframscript returned non-zero exit code {proc.returncode}")

# parse lambda line (as before)
lambda_line = None
for line in proc.stdout.splitlines():
    line = line.strip()
    if line.startswith("PY_LAMBDA="):
        lambda_line = line[len("PY_LAMBDA="):].strip()
        break

if lambda_line is None:
    raise RuntimeError("PY_LAMBDA not found. Full stdout:\n" + proc.stdout)

print("Raw lambda_line:", lambda_line)

# --- robust unquoting ---
import ast, re, math

# find the colon separating "lambda y" and the body
m = re.match(r'^(lambda\s+\w+\s*:\s*)(.*)$', lambda_line)
if not m:
    raise RuntimeError("Parsed PY_LAMBDA doesn't look like a lambda: " + lambda_line)

lambda_head = m.group(1)   # e.g. "lambda y: "
lambda_body_raw = m.group(2).strip()

# If body is a Python string literal (starts and ends with matching quotes),
# unquote it with ast.literal_eval to get the inner text.
if (lambda_body_raw.startswith('"') and lambda_body_raw.endswith('"')) or \
   (lambda_body_raw.startswith("'") and lambda_body_raw.endswith("'")):
    # safe unquote
    try:
        lambda_body = ast.literal_eval(lambda_body_raw)
    except Exception as e:
        raise RuntimeError(f"Failed to literal_eval Mathematica string: {e}\n{lambda_body_raw}")
else:
    lambda_body = lambda_body_raw

# now compose final lambda string
final_lambda_str = lambda_head + lambda_body
print("Final lambda string to eval:", final_lambda_str)

# safe eval context
safe_globals = {"math": math}
f = eval(final_lambda_str, safe_globals)

# test
for y in (0.0, 0.5, 1.0, 1.570795):
    print(f"f({y}) = {f(y)}")
