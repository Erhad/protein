#!/bin/bash
PDF=$1
TEX=$2
CSV=$3

export PATH="$PATH:/Library/TeX/texbin"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
PAGES=$(pdfinfo "$PDF" 2>/dev/null | awk '/^Pages:/ {print $2}')
WORDS=$(texcount -sum -1 -inc "$TEX" 2>/dev/null | head -1 | tr -d ' ')

# Write header if file doesn't exist
if [ ! -f "$CSV" ]; then
    echo "timestamp,pages,words" > "$CSV"
fi

echo "$TIMESTAMP,$PAGES,$WORDS" >> "$CSV"
