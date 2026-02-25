import sys
import os
import traceback

# Monkeypatch sys.exit and os._exit to print a python traceback
_orig_sys_exit = sys.exit
_orig_os_exit = os._exit

def new_sys_exit(*args, **kwargs):
    print("SYS.EXIT CALLED. TRACEBACK:", flush=True)
    traceback.print_stack()
    _orig_sys_exit(*args, **kwargs)

def new_os_exit(*args, **kwargs):
    print("OS._EXIT CALLED. TRACEBACK:", flush=True)
    traceback.print_stack()
    _orig_os_exit(*args, **kwargs)

sys.exit = new_sys_exit
os._exit = new_os_exit

# Run the real script
sys.argv.extend(['--epochs', '1', '--max-steps', '1', '--batch-size', '1'])
import pathlib
sys.path.append('scripts')
import train_m1_t5
args = train_m1_t5._parse_args()
trainer = train_m1_t5.T5Trainer(args.model, pathlib.Path(args.data_dir), pathlib.Path(args.output), 1, 1, args.lr, 1, 256, 256)
trainer.run()
