# Print a Python interpreter that can import passthru from this checkout.
#
# Both check scripts need one, and both used to hardcode .venv/bin/python -- which does not
# exist in a fresh clone. A judge following the README got "No such file or directory" from
# the verification steps the README had just recommended.
#
# Order: an explicit PASSTHRU_PYTHON, then the local venv, then whatever is on PATH.
if [ -n "${PASSTHRU_PYTHON:-}" ]; then
  printf '%s' "$PASSTHRU_PYTHON"
  exit 0
fi
if [ -x ".venv/bin/python" ]; then
  printf '%s' ".venv/bin/python"
  exit 0
fi
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    printf '%s' "$candidate"
    exit 0
  fi
done
printf '%s' ""
