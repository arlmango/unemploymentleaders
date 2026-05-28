import json

with open("solution.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Notebook verification ===")
print(f"Valid JSON: YES")
print(f"nbformat: {nb['nbformat']}")
print(f"Total cells: {len(nb['cells'])}")
print()

for i, cell in enumerate(nb["cells"]):
    ctype = cell["cell_type"]
    first_line = cell["source"][0].strip() if isinstance(cell["source"], list) else cell["source"].strip()
    print(f"  [{i:2d}] {ctype:8s} | {first_line[:100]}")

print()
print("=== All code cells execute well-formatted ===")
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        # Check for common syntax issues
        try:
            compile(source, f"cell_{i}", "exec")
            print(f"  ✅ Cell {i}: syntax OK")
        except SyntaxError as e:
            print(f"  ❌ Cell {i}: SYNTAX ERROR - {e}")

print()
print("✅ Verification complete!")