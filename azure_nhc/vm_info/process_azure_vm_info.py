#!/usr/bin/env python3

import json as js

js_filename = "azure_vm_info.json"
js_file = open(js_filename)
vm_data = js.load(js_file)

tmp_data = {}
for vm in vm_data:
    print(vm["name"])
    if "resourceType" in vm and vm["resourceType"] == "virtualMachines":
        if vm["name"] not in tmp_data:
            tmp_data[vm["name"]] = vm

for vm_size in tmp_data:
    print("{} : {}".format(tmp_data[vm_size]["family"], tmp_data[vm_size]["name"]))

with open("azure_vm_info_processed.json", "w") as outfile:
    js.dump(tmp_data, outfile, indent=4)
