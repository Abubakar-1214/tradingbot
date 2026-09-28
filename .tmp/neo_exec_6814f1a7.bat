python -c "
import re
p=r'e:\Desktop\NeoMind\Bazz\autonoumuse_trader\models_audit_report.md'
lines=open(p,encoding='utf-8',errors='replace').read().splitlines()
print('TOTAL LINES:', len(lines))
print('--- HEADERS ---')
for i,l in enumerate(lines,1):
    if l.startswith('#'):
        print(i, l[:100])
"