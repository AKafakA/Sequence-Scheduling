import glob
import json

len_files = glob.glob("./sharegpt-val-10k-*-length.json")
output_list = []

for f in len_files:
    with open(f, "rb") as infile:
        output_list.extend(json.load(infile))
id_sets = {}
deduplicated_list = []
for item in output_list:
    item_id = item["id"]
    if item_id not in id_sets:
        id_sets[item_id] = 1
        deduplicated_list.append(item)
print(len(deduplicated_list))
newlist = sorted(deduplicated_list, key=lambda d: d['id'])
with open('./sharegpt-val-10k-length.json', 'w') as fp:
    json.dump(newlist, fp)




