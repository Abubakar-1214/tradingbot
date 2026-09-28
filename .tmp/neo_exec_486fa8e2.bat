python -c "
import re
p=r'e:\Desktop\NeoMind\Bazz\autonoumuse_trader\models_audit_report.md'
txt=open(p,encoding='utf-8',errors='replace').read()
refs=re.findall(r'arXiv:\s*[\d.]+|arxiv\.org/abs/[\d.]+|arXiv:\d{4}\.\d{4,5}', txt)
print('arxiv ref count:', len(refs))
for r in refs[:20]: print(r)
print('--- Roman-Urdu check ---')
for kw in ['Roman','Urdu','اردو','summary']:
    print(kw, txt.lower().count(kw.lower()))
"