import os
from glob import glob
from sklearn.model_selection import train_test_split

# ======================================================================================================
# data structure
# ======================================================================================================
# cross-subject(subject id) => train: 3, 4, 6, 7, 9, 12, 13, 15, 17, 19, 25 | test: remaining 7 subjects
# cross-view1(camera id) => train: 1 | validation: 5 | test: 2
# cross-view2(camera id) => train: 1, 3, 4, 6, 7 | validation: 5 | test: 2
#
# csv header => sub directory file path, index, category
# official site: https://project.inria.fr/toyotasmarthome/
# 
# *files root path means is bellow
# ex)
# json (files_root_path)
#    L xxx.json
#    L xxx.json
#        .
#        .
#        .
# ======================================================================================================

def cross_subject(files_root_path: str, save_path: str):
    # path check
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # read a files path(root path of the toyotasmarthome files)
    files_path = glob(os.path.join(files_root_path, "*"))

    trains = []; tests = []; activitynames = []
    for file_path in files_path:
        filename = file_path.split("/")[-1]
        
        # Activityname_p[id]_r[XX]_[XX]_c[0-7]
        splited_filename = filename.split("_")

        # get a activityname from splited filename
        activityname = splited_filename[0]

        # get a subject id from splited fimename
        subject_id = int(splited_filename[1][1:])

        # for indexing
        if not activityname in activitynames:
            activitynames.append(activityname)
        
        # making label
        label = "{},{},{}".format(filename, activitynames.index(activityname), activityname)

        if subject_id in [3, 4, 6, 7, 9, 12, 13, 15, 17, 19, 25]:
            trains.append(label) # train
        else:
            tests.append(label) # test
    
    print(f"[cross subject] train: {len(trains)}, test: {len(tests)}, categories: {len(activitynames)}")

    # save
    with open(os.path.join(save_path, "train.csv"), "w") as f:
        f.writelines("\n".join(trains))
    with open(os.path.join(save_path, "test.csv"), "w") as f:
        f.writelines("\n".join(tests))

def cross_view1(files_root_path: str, save_path: str):
    # path check
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # read a files path(root path of the toyotasmarthome files)
    files_path = glob(os.path.join(files_root_path, "*"))

    trains = []; vals = []; tests = []; activitynames = []
    for file_path in files_path:
        filename = file_path.split("/")[-1]

        # Activityname_p[id]_r[XX]_[XX]_c[0-7]
        splited_filename = filename.split("_")

        # get a activityname from splited filename
        activityname = splited_filename[0]

        # get a camera id from splited filename
        camera_id = int(splited_filename[4][1:])

        if camera_id in [1, 2, 5]:
            # for indexing
            if not activityname in activitynames:
                activitynames.append(activityname)
        
            # making label
            label = "{},{},{}".format(filename, activitynames.index(activityname), activityname)

            # train
            if camera_id == 1:
                trains.append(label)

            # validation
            if camera_id == 5:
                vals.append(label)

            # test
            if camera_id == 2:
                tests.append(label)
    
    print(f"[cross view 1] train: {len(trains)}, test: {len(tests)}, val: {len(vals)}, categories: {len(activitynames)}")

    # save
    with open(os.path.join(save_path, "train.csv"), "w") as f:
        f.writelines("\n".join(trains))
    with open(os.path.join(save_path, "val.csv"), "w") as f:
        f.writelines("\n".join(vals))
    with open(os.path.join(save_path, "test.csv"), "w") as f:
        f.writelines("\n".join(tests))

def cross_view2(files_root_path: str, save_path: str):
    # path check
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # read a files path(root path of the toyotasmarthome files)
    files_path = glob(os.path.join(files_root_path, "*"))

    trains = []; vals = []; tests = []; activitynames = []
    for file_path in files_path:
        filename = file_path.split("/")[-1]

        # Activityname_p[id]_r[XX]_[XX]_c[0-7]
        splited_filename = filename.split("_")

        # get a activityname from splited filename
        activityname = splited_filename[0]

        # get a camera id from splited filename
        camera_id = int(splited_filename[4][1:])

        if camera_id in [1, 2, 3, 4, 5, 6, 7]:
            # for indexing
            if not activityname in activitynames:
                activitynames.append(activityname)
        
            # making label
            label = "{},{},{}".format(filename, activitynames.index(activityname), activityname)

            # train
            if camera_id in [1, 3, 4, 6, 7]:
                trains.append(label)

            # validation
            if camera_id == 5:
                vals.append(label)

            # test
            if camera_id == 2:
                tests.append(label)
    
    print(f"[cross view 1] train: {len(trains)}, test: {len(tests)}, val: {len(vals)}, categories: {len(activitynames)}")

    # save
    with open(os.path.join(save_path, "train.csv"), "w") as f:
        f.writelines("\n".join(trains))
    with open(os.path.join(save_path, "val.csv"), "w") as f:
        f.writelines("\n".join(vals))
    with open(os.path.join(save_path, "test.csv"), "w") as f:
        f.writelines("\n".join(tests))

def custom_split(files_root_path: str, save_path: str, shuffle: bool, test_size: float):
    # path check
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # read a files path(root path of the toyotasmarthome files)
    files_path = glob(os.path.join(files_root_path, "*"))

    datas_x = []; datas_y = []; activitynames = []
    for file_path in files_path:
        filename = file_path.split("/")[-1]

        # Activityname_p[id]_r[XX]_[XX]_c[0-7]
        splited_filename = filename.split("_")

        # get a activityname from splited filename
        activityname = splited_filename[0]

        # for indexing
        if not activityname in activitynames:
            activitynames.append(activityname)
        index = activitynames.index(activityname)

        # making label
        label = "{},{},{}".format(filename, index, activityname)

        # append
        datas_x.append(label)
        datas_y.append(index)

    trains, tests, _, _ = train_test_split(datas_x, datas_y, shuffle=shuffle, test_size=test_size)
    
    print(f"[custom split] train: {len(trains)}, test: {len(tests)}, categories: {len(activitynames)}, test size: {test_size}")

    # save
    with open(os.path.join(save_path, "train.csv"), "w") as f:
        f.writelines("\n".join(trains))
    with open(os.path.join(save_path, "test.csv"), "w") as f:
        f.writelines("\n".join(tests))