import json


with open("xd.json", "r") as file:
    imgs = json.load(file)

print(imgs)
pts = 0
for file in imgs:

    name = file.strip(".jpg")
    print(f"num: {len(name)}, {len(imgs[file])}")
    decoded = imgs[file]
    print(name, imgs[file])
    is_all = True
    for let_name, let_dec in zip(name, decoded):
        if let_name != let_dec or len(name) != len(imgs[file]):
            print(let_dec, let_name)
        if let_name == let_dec:
            pts += 1
        else:
            is_all = False
    if is_all and len(name) == len(imgs[file]):
        pts += 3

print(pts)
        