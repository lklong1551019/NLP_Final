#!/bin/bash
# Build the paper PDF and report the page count against the 4-page limit.
#
#   bash paper/build.sh
#
# Regenerates the tables first, so the PDF can never report stale numbers.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Regenerating tables =="
if [ -d results ] && [ -n "$(ls -A results 2>/dev/null || true)" ]; then
    python scripts/export_paper_tables.py
else
    echo "No results yet; using placeholders."
    python scripts/export_paper_tables.py --placeholders
fi

cd paper

if ! command -v pdflatex >/dev/null 2>&1; then
    echo ""
    echo "pdflatex not found. Either install TeX Live / MiKTeX, or upload this"
    echo "directory to Overleaf, which needs no local install."
    exit 1
fi

echo ""
echo "== Building =="
# bibtex needs a first pass to see the citations, then two more to settle refs.
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null 2>&1 || echo "(bibtex reported problems -- check main.blg)"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

if [ ! -f main.pdf ]; then
    echo "Build failed. See paper/main.log"
    exit 1
fi

echo ""
if command -v pdfinfo >/dev/null 2>&1; then
    pages=$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')
    echo "Built main.pdf -- $pages pages total."
    echo "The limit is 4 pages of content; references, Limitations, Ethical"
    echo "Considerations and appendices do not count toward it."
else
    echo "Built main.pdf"
fi

echo ""
echo "Remaining TODOs:"
grep -c "TODO" main.tex | xargs printf "  %s in main.tex\n"
grep -rc "TODO" generated/*.tex 2>/dev/null | grep -v ":0$" | sed 's/^/  /' || true
