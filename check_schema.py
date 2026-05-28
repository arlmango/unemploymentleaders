import pyarrow.parquet as pq

# Check business cards
table = pq.read_table("business_cards_MDQ.parquet")
print("=== business_cards_MDQ.parquet columns ===")
for field in table.schema:
    print(f"  '{field.name}' : {field.type}")

print()

# Check consumer cards
table2 = pq.read_table("consumer_cards_MDQ.parquet")
print("=== consumer_cards_MDQ.parquet columns ===")
for field in table2.schema:
    print(f"  '{field.name}' : {field.type}")

print()

# Check merchants reference
table3 = pq.read_table("merchants_reference.parquet")
print("=== merchants_reference.parquet columns ===")
for field in table3.schema:
    print(f"  '{field.name}' : {field.type}")