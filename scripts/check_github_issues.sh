#!/bin/bash
# Claude Code SessionStart hook: surface open GitHub issues
# Output is injected into Claude's context at conversation start

ISSUES=$(gh issue list --state open --limit 10 --json number,title,labels,createdAt 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "GitHub issues: unable to fetch (gh CLI not authenticated or offline)"
    exit 0
fi

COUNT=$(echo "$ISSUES" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)

if [ "$COUNT" = "0" ] || [ -z "$COUNT" ]; then
    echo "GitHub issues: 0 open"
else
    echo "GitHub issues: $COUNT open"
    echo "$ISSUES" | python3 -c "
import sys, json
issues = json.load(sys.stdin)
for i in issues:
    labels = ', '.join(l['name'] for l in i.get('labels', []))
    label_str = f' [{labels}]' if labels else ''
    print(f\"  #{i['number']}: {i['title']}{label_str}\")
" 2>/dev/null
fi
