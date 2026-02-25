import sys
import traceback

sys.argv.extend(['--epochs', '1', '--max-steps', '1'])

try:
    with open('scripts/train_m1_t5.py', 'r', encoding='utf-8') as f:
        code = compile(f.read(), 'scripts/train_m1_t5.py', 'exec')
        exec(code, {'__name__': '__main__'})
except Exception as e:
    print('CRASH TRACEBACK:', flush=True)
    traceback.print_exc()
    sys.exit(1)
