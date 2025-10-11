# scripts/run_webui.py
#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.ui.webui import main

if __name__ == "__main__":
    main()