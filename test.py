# labeler testing...
from Labeler import cross_subject, cross_view1, cross_view2, custom_split

cross_subject("../Data/ToyotaSmartHome_frames/", "./labels/cross_subject/")
cross_view1("../Data/ToyotaSmartHome_frames/", "./labels/cross_view1/")
cross_view2("../Data/ToyotaSmartHome_frames/", "./labels/cross_view2/")
custom_split("../Data/ToyotaSmartHome_frames/", "./labels/custom_split_7_3/", shuffle=True, test_size=0.3)

