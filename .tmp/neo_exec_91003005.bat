cd /d e:\Desktop\NeoMind\Bazz\autonoumuse_trader && python -c "
import os
files=['models/transformer_policy.py','models/meta_learning.py','models/mcts.py','models/adversarial_training.py','models/dreamer_agent.py','models/ensemble.py','models/position_sizing.py','models/risk_supervisor.py','models/dreamer_components.py']
for f in files:
    print(f, os.path.getsize(f) if os.path.exists(f) else 'MISSING', sum(1 for _ in open(f,encoding='utf-8',errors='replace')) if os.path.exists(f) else '')
"