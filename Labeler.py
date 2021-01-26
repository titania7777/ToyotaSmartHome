import os
from glob import glob
from sklearn.model_selection import train_test_split

# ======================================================================================================
# data structure
# ======================================================================================================
# cross-subject(subject id) => train: 3, 4, 6, 7, 9, 12, 13, 15, 17, 19, 25 | test: remaining 7 subjects
# cross-view1(camera id) => train: 1 | validation: 5 | test: 2
# cross-view2(camera id) => train: 1, 3, 4, 6, 7 | validation: 5 | test: 2
# csv header: sub directory file path, index, category
# official site: https://project.inria.fr/toyotasmarthome/
# ======================================================================================================

def cross_subset(videos_path: str, save_path: str):
    videos_path = glob(os.path.join(videos_root_path, "*"))

    trains = []
    tests = []
    categories = []
    for video_path in videos_path:
        filename = video_path.split("/")[-1][:-4] # all of videos format is mp4
        splited_filename = filename.split("_")

        # get category from splited filename
        category = splited_filename[0]

        # for indexing
        if not category in categories:
            categories.append(category)
        
        # get remain informations from splited filename
        subset_id = int(splited_filename[1][1:])
        camera_id = int(splited_filename[4][1:])
        
        # making label
        label = "{},{},{}".format(filename, categories.index(category), category)

        # [cross subset]
        # train: 10682, test: 5433
        if subset_id in [3, 4, 6, 7, 9, 12, 13, 15, 17, 19, 25]:
            trains.append(label)
        else:
            tests.append(label)

    # save
    with open(os.path.join(save_path, "train.csv"), "w") as f:
        f.writelines("\n".join(trains))
    with open(os.path.join(save_path, "test.csv"), "w") as f:
        f.writelines("\n".join(tests))

def cross_view1(videos_root_path: str, save_path: str) -> None:
    videos_path = glob(os.path.join(videos_root_path, "*"))

    trains = []
    vals = []
    tests = []
    categories = []
    for video_path in videos_path:
        filename = video_path.split("/")[-1][:-4] # all of videos format is mp4
        splited_filename = filename.split("_")

        # get category from splited filename
        category = splited_filename[0]

        # for indexing
        if not category in categories:
            categories.append(category)
        
        # get remain informations from splited filename
        subset_id = int(splited_filename[1][1:])
        camera_id = int(splited_filename[4][1:])
        
        # making label
        label = "{},{},{}".format(filename, categories.index(category), category)

        # validation for cross view
        if camera_id == 5:
            vals.append(label)

        # test for cross view
        if camera_id == 2:
            tests.append(label)

        # [cross view 1]
        # activities in dining room
        # train: 1877, val: 4188, test: 1901
        if camera_id == 1:
            trains.append(label)

    # save
    with open(os.path.join(save_path, "train.csv"), "w") as f:
        f.writelines("\n".join(trains))
    with open(os.path.join(save_path, "val.csv"), "w") as f:
        f.writelines("\n".join(vals))
    with open(os.path.join(save_path, "test.csv"), "w") as f:
        f.writelines("\n".join(tests))

def cross_view2(videos_root_path: str, save_path: str) -> None:
    videos_path = glob(os.path.join(videos_root_path, "*"))

    trains = []
    vals = []
    tests = []
    categories = []
    for video_path in videos_path:
        filename = video_path.split("/")[-1][:-4] # all of videos format is mp4
        splited_filename = filename.split("_")

        # get category from splited filename
        category = splited_filename[0]

        # for indexing
        if not category in categories:
            categories.append(category)
        
        # get remain informations from splited filename
        subset_id = int(splited_filename[1][1:])
        camera_id = int(splited_filename[4][1:])
        
        # making label
        label = "{},{},{}".format(filename, categories.index(category), category)

        # validation for cross view
        if camera_id == 5:
            vals.append(label)

        # test for cross view
        if camera_id == 2:
            tests.append(label)

        # [cross view 1]
        # activities in dining room
        # train: 1877, val: 4188, test: 1901
        if camera_id in [1, 3, 4, 6, 7]:
            trains.append(label)

    # save
    with open(os.path.join(save_path, "train.csv"), "w") as f:
        f.writelines("\n".join(trains))
    with open(os.path.join(save_path, "val.csv"), "w") as f:
        f.writelines("\n".join(vals))
    with open(os.path.join(save_path, "test.csv"), "w") as f:
        f.writelines("\n".join(tests))

def custom_split(videos_root_path: str, save_path: str, shuffle: bool, test_size: float) -> None:
    videos_path = glob(os.path.join(videos_root_path, "*"))

    datas_x = []
    datas_y = []
    categories = []
    for video_path in videos_path:
        filename = video_path.split("/")[-1][:-4] # all of videos format is mp4
        splited_filename = filename.split("_")

        # get category from splited filename
        category = splited_filename[0]

        # for indexing
        if not category in categories:
            categories.append(category)
        index = categories.index(category)

        # get remain informations from splited filename
        subset_id = int(splited_filename[1][1:])
        camera_id = int(splited_filename[4][1:])

        # making label
        label = "{},{},{}".format(filename, index, category)

        # [custom split]
        datas_x.append(label)
        datas_y.append(index)

    trains, tests, _, _ = train_test_split(datas_x, datas_y, shuffle=shuffle, test_size=test_size)

    # save
    with open(os.path.join(save_path, "train.csv"), "w") as f:
        f.writelines("\n".join(trains))
    with open(os.path.join(save_path, "test.csv"), "w") as f:
        f.writelines("\n".join(tests))