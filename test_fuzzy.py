from rapidfuzz import fuzz

name_a = "ill mbps with wan lan ip"
name_b = "service description ill connectivity mbps"

token_set = fuzz.token_set_ratio(name_a, name_b) / 100.0
partial = fuzz.partial_ratio(name_a, name_b) / 100.0
ratio = fuzz.ratio(name_a, name_b) / 100.0

print(f"token_set_ratio: {token_set:.3f}")
print(f"partial_ratio: {partial:.3f}")
print(f"ratio: {ratio:.3f}")
