import ast
import sys

def verify_fixes():
    try:
        with open('pages/1_Cockpit.py', 'r') as f:
            code = f.read()
            tree = ast.parse(code)
    except FileNotFoundError:
        print("ERROR: pages/1_Cockpit.py not found.")
        sys.exit(1)

    # 1. Verify P&L help text
    expected_help = "Total change in account equity since prior day close (as reported by IBKR)."
    found_help = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'pnl_help':
                    if isinstance(node.value, (ast.Constant, ast.Str)):
                        val = node.value.value if isinstance(node.value, ast.Constant) else node.value.s
                        if val == expected_help:
                            found_help = True
                            print("✅ P&L help text updated correctly.")
                        else:
                            print(f"❌ P&L help text mismatch. Found: '{val}'")

    if not found_help:
        print("❌ Could not find 'pnl_help' assignment with correct text.")

    # 2. Verify quantity formatting
    # Looking for f-string format specifier :g
    found_qty_fmt = False

    # We are looking for f"• {symbol}: {qty:g}"
    # In AST, this is a JoinedStr with formatted values
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            # Check components of the f-string
            # We expect a Constant "• ", a FormattedValue (symbol), a Constant ": ", and a FormattedValue (qty)
            # The qty FormattedValue should have format_spec equal to a JoinedStr containing "g"

            # Simplified check: just look for a FormattedValue with format_spec "g"
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    if value.format_spec:
                        # format_spec is a JoinedStr
                        if len(value.format_spec.values) == 1 and isinstance(value.format_spec.values[0], (ast.Constant, ast.Str)):
                            spec = value.format_spec.values[0].value if isinstance(value.format_spec.values[0], ast.Constant) else value.format_spec.values[0].s
                            if spec == 'g':
                                found_qty_fmt = True
                                print("✅ Quantity formatting using ':g' found.")
                                break
        if found_qty_fmt:
            break

    if not found_qty_fmt:
        print("❌ Could not find quantity formatting with ':g'.")

    if found_help and found_qty_fmt:
        print("SUCCESS: All fixes verified.")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    verify_fixes()
