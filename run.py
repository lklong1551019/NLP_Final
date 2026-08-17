#!/usr/bin/env python3
"""CLI entry point.

    python run.py --config configs/xcopa_vi_qwen_deepseek.yaml
    python run.py --config configs/xcopa_vi_qwen_deepseek.yaml --end 5
    python run.py --list
"""

from faithlm.run import main

if __name__ == "__main__":
    main()
