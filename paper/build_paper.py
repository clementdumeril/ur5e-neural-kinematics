"""
Rebuild the paper: figures, numbers, then the PDF.

    python paper/build_paper.py

Needs a LaTeX distribution with latexmk (tested with MiKTeX 25.12).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

for script in ('make_figures.py', 'make_numbers.py'):
    subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)
subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
               cwd=HERE, check=True)
subprocess.run(['latexmk', '-c', 'main.tex'], cwd=HERE, check=True)
print(os.path.join(HERE, 'main.pdf'))
