cd /d e:\Desktop\NeoMind\Bazz\autonoumuse_trader && python -c "
lines=open('models/transformer_policy.py',encoding='utf-8',errors='replace').read().splitlines()
for i in range(348,356):
    if i<len(lines): print(i+1, lines[i])
print('---META 165-172---')
lines=open('models/meta_learning.py',encoding='utf-8',errors='replace').read().splitlines()
for i in range(164,172):
    if i<len(lines): print(i+1, lines[i])
print('---META 247-260---')
for i in range(246,260):
    if i<len(lines): print(i+1, lines[i])
"