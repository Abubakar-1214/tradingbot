python -c "
import re, html
with open('research/tradingbot_audit_report.html', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()
# Strip scripts/styles
content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL|re.IGNORECASE)
content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL|re.IGNORECASE)
# Convert common tags to newlines/markers
content = re.sub(r'<(h1|h2|h3|h4|p|li|tr|div|br|td|th)[^>]*>', '\n', content, flags=re.IGNORECASE)
content = re.sub(r'</(h1|h2|h3|h4|p|li|tr|div|td|th)>', '\n', content, flags=re.IGNORECASE)
content = html.unescape(content)
content = re.sub(r'<[^>]+>', ' ', content)
lines = [l.strip() for l in content.split('\n')]
lines = [l for l in lines if l]
text = '\n'.join(lines)
with open('research/_audit_report_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print('TOTAL CHARS:', len(text))
print('TOTAL LINES:', len(lines))
"