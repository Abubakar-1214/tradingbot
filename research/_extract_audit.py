# -*- coding: utf-8 -*-
"""Extract readable text from tradingbot_audit_report.html"""
import re
import html
import os

base = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(base, "tradingbot_audit_report.html")
dst = os.path.join(base, "_audit_report_text.txt")

with open(src, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Strip scripts/styles
content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)

# Convert block tags to newlines
content = re.sub(r"<(h1|h2|h3|h4|h5|p|li|tr|div|br|td|th|section|article|pre)[^>]*>", "\n", content, flags=re.IGNORECASE)
content = re.sub(r"</(h1|h2|h3|h4|h5|p|li|tr|div|td|th|section|article|pre)>", "\n", content, flags=re.IGNORECASE)

content = html.unescape(content)
content = re.sub(r"<[^>]+>", " ", content)

lines = [l.strip() for l in content.split("\n")]
lines = [l for l in lines if l]
text = "\n".join(lines)

with open(dst, "w", encoding="utf-8") as f:
    f.write(text)

print("TOTAL CHARS:", len(text))
print("TOTAL LINES:", len(lines))
