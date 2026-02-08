import os
import sys
from defusedxml import ElementTree as ET

min_cov = float(os.environ.get("COVERAGE_MIN", "90"))
path = os.environ.get("COVERAGE_XML", "../coverage/cobertura.xml")

try:
    tree = ET.parse(path)
except FileNotFoundError:
    print(f"Coverage report not found: {path}", file=sys.stderr)
    raise SystemExit(2)
except ET.ParseError as e:
    print(f"Could not parse coverage XML: {path}: {e}", file=sys.stderr)
    raise SystemExit(2)

root = tree.getroot()
rate_attr = root.attrib.get("line-rate")
if rate_attr is None:
    print("Coverage XML missing 'line-rate' attribute on root element", file=sys.stderr)
    raise SystemExit(2)

rate = float(rate_attr) * 100.0
if rate + 1e-9 < min_cov:
    print(f"Coverage {rate:.2f}% is below minimum {min_cov:.2f}%")
    raise SystemExit(1)

print(f"Coverage {rate:.2f}% meets minimum {min_cov:.2f}%")
