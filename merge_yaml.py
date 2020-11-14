import yaml
import copy

with open('/home/zexin/project/mice/algorithm/unsupervised_classification/hierarchical_clustering/environment.yaml', 'r') as fin:
    newYaml = yaml.safe_load(fin)
with open('./environment.yml') as fin:
    oldYaml = yaml.safe_load(fin)


yamlMerge = copy.deepcopy(oldYaml)

for pkg in newYaml['dependencies'][-1]['pip']:
    if pkg not in yamlMerge['dependencies'][-1]['pip']:
        # print(pkg)
        yamlMerge['dependencies'][-1]['pip'].append(pkg)
for pkg in newYaml['dependencies']:
    if isinstance(pkg, str) and pkg not in yamlMerge['dependencies']:
        yamlMerge['dependencies'].append(pkg)


tmp = [pkg for pkg in yamlMerge['dependencies'] if isinstance(pkg, str)]
tmp1 = [pkg for pkg in yamlMerge['dependencies'] if isinstance(pkg, dict)]
tmp1[0]['pip'].sort()
tmp.sort()
yamlMerge['dependencies'] = tmp + tmp1

print(tmp)
print(tmp1)

with open('./environment.yaml','w') as fout:
    yaml.dump( yamlMerge, fout)


