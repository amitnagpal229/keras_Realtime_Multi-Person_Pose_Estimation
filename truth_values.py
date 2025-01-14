import numpy as np

truth_labels = np.array([
    [1081, 530, 1, 1],
    [95, 445, 2, 1],
    [1209, 329, 2, 1],
    [1120, 565, 2, 1],
    [1124, 583, 2, 1],
    [1123, 614, 2, 1],
    [1920, 0, 2, 1],
    [1244, 196, 2, 1],
    [1242, 202, 2, 1],
    [0, 0, 2, 1],
    [1300, 361, 2, 1],
    [1302, 359, 2, 1],
    [1295, 387, 2, 1],
    [1310, 123, 2, 1],
    [408, 323, 1, 1],
    [187, 188, 1, 1],
    [691, 305, 2, 1],
    [674, 307, 2, 1],
    [660, 304, 2, 1],
    [650, 297, 2, 1],
    [382, 425, 2, 1],
    [643, 364, 1, 1],
    [632, 358, 1, 1],
    [263, 455, 1, 1],
    [119, 237, 1, 1],
    [1016, 77, 2, 1],
    [1051, 336, 2, 1],
    [1056, 330, 2, 1],
    [1067, 328, 2, 1],
    [1066, 122, 2, 1],
    [842, 268, 2, 1],
    [876, 284, 2, 1],
    [911, 310, 2, 1],
    [419, 194, 1, 1],
    [194, 605, 2, 1],
    [918, 255, 2, 1],
    [1087, 288, 2, 1],
    [1071, 260, 2, 1],
    [1061, 241, 2, 1],
    [1044, 207, 2, 1],
    [1036, 195, 2, 1],
    [148, 605, 2, 1],
    [142, 600, 2, 1],
    [136, 619, 2, 1],
    [137, 659, 2, 1],
    [418, 532, 2, 1],
    [416, 535, 2, 1],
    [417, 557, 2, 1],
    [415, 596, 2, 1],
    [89, 366, 2, 1],
    [850, 279, 2, 1],
    [729, 301, 2, 1],
    [735, 306, 2, 1],
    [757, 64, 2, 1],
    [962, 359, 2, 1],
    [970, 388, 2, 1],
    [1146, 306, 2, 1],
    [1148, 320, 2, 1],
    [1146, 341, 2, 1],
    [506, 320, 1, 1],
    [473, 311, 1, 1],
    [445, 318, 1, 1],
    [399, 312, 1, 1],
    [336, 338, 1, 1],
    [942, 356, 1, 1],
    [228, 217, 1, 1],
    [344, 413, 1, 1],
    [253, 189, 1, 1],
    [296, 214, 1, 1],
    [180, 307, 1, 1],
    [135, 313, 1, 1],
    [780, 500, 1, 1],
    [246, 285, 1, 1],
    [240, 274, 1, 1],
    [48, 317, 1, 1],
    [1068, 207, 2, 1],
    [1070, 196, 2, 1],
    [1071, 199, 2, 1],
    [1053, 192, 2, 1],
    [1081, 241, 2, 1],
    [1112, 206, 2, 1],
    [953, 158, 2, 1],
    [956, 150, 2, 1],
    [1006, 152, 2, 1],
    [1398, 189, 2, 1],
    [1399, 231, 2, 1],
    [359, 321, 1, 1],
    [332, 151, 1, 1],
    [434, 69, 1, 1],
    [305, 386, 1, 1],
    [303, 228, 1, 1],
    [285, 247, 1, 1],
    [278, 265, 1, 1],
    [252, 214, 1, 1],
    [248, 184, 1, 1],
    [224, 201, 1, 1],
    [309, 243, 1, 1],
    [829, 151, 2, 1],
    [828, 153, 2, 1],
    [823, 171, 2, 1],
    [821, 197, 2, 1],
    [1281, 355, 2, 1],
    [1343, 180, 2, 1],
    [1011, 299, 2, 1],
    [971, 350, 2, 1]
])

jump_truth = np.array([
    [-63, -123, 1, 1],
    [-52, -56, 2, 1],
    [218, -218, 2, 1],
    [87, 32, 2, 1],
    [0, -104, 2, 1],
    [-27, -31, 2, 1],
    [-11, -21, 2, 1],
    [0, 0, 1, 1],
    [-69, -88, 2, 1],
    [44, -150, 2, 1],
    [0, 0, 2, 1],
    [0, -147, 2, 1],
    [-29, -35, 2, 1],
    [0, 0, 2, 1],
    [0, -35, 1, 1],
    [-65, -72, 2, 1],
    [-28, -33, 2, 1],
    [182, -183, 2, 1],
    [10, 0, 2, 1],
    [6, -183, 2, 1],
    [-32, -40, 2, 1],
    [0, 0, 2, 1],
    [-34, -80, 1, 1],
    [-71, -84, 1, 1],
    [-63, -84, 1, 1],
    [-223, -232, 2, 1],
    [-45, -49, 2, 1],
    [49, -84, 2, 1],
    [15, -8, 2, 1],
    [0, -64, 2, 1],
    [-33, -45, 2, 1],
    [0, 0, 2, 1],
    [0, 0, 1, 1],
    [-179, -189, 2, 1],
    [0, -117, 2, 1],
    [-15, -31, 2, 1],
    [142, -72, 2, 1],
    [-42, -49, 2, 1],
    [0, 0, 2, 1],
    [-26, -75, 1, 1],
    [-59, -93, 1, 1],
    [-55, -66, 2, 1],
    [88, -85, 2, 1],
    [1, -4, 2, 1],
    [0, -67, 2, 1],
    [-28, -38, 2, 1],
    [0, 0, 2, 1],
    [0, 0, 1, 1],
    [-121, -136, 2, 1],
    [0, -66, 2, 1],
    [0, 0, 2, 1],
    [198, -192, 2, 1],
    [-26, -32, 2, 1],
    [0, 0, 2, 1],
    [-34, -77, 1, 1],
    [-63, -89, 1, 1],
    [-77, -91, 1, 1],
    [-38, -91, 1, 1],
    [-45, -153, 2, 1],
    [67, -55, 2, 1],
    [-8, -11, 2, 1],
    [-9, -73, 2, 1],
    [-27, -35, 2, 1],
    [0, 0, 2, 1],
    [-44, -54, 1, 1],
    [-198, -204, 2, 1],
    [-31, -48, 2, 1],
    [28, 6, 2, 1],
    [0, -66, 2, 1],
    [-26, -33, 2, 1],
    [0, 0, 2, 1],
    [-56, -90, 1, 1],
    [-65, -69, 2, 1],
    [0, -56, 2, 1],
    [0, 0, 2, 1],
    [0, -140, 2, 1],
    [-26, -35, 2, 1],
    [0, 0, 2, 1],
    [-32, -51, 2, 1],
    [-1, -66, 2, 1],
    [0, 0, 2, 1],
    [7, -76, 2, 1],
    [-28, -30, 2, 1],
    [0, 0, 2, 1],
    [-66, -98, 1, 1],
    [-94, -98, 1, 1],
    [-65, -98, 1, 1],
    [-33, -98, 1, 1],
    [-117, -123, 2, 1],
    [43, -98, 2, 1],
    [74, 0, 2, 1],
    [0, -66, 2, 1],
    [0, 0, 2, 1],
    [-62, -81, 2, 1],
    [193, -163, 2, 1],
    [539, 502, 2, 1],
    [1, -90, 2, 1],
    [-28, -31, 2, 1],
    [0, 0, 2, 1],
    [-31, -67, 1, 1],
    [-51, -67, 2, 1],
    [-32, -36, 2, 1],
    [0, -51, 2, 1],
    [24, 0, 2, 1],
    [120, -106, 2, 1],
    [-36, -47, 2, 1],
    [-9, -15, 2, 1],
    [-31, -105, 1, 1],
    [-71, -123, 1, 1],
    [-85, -101, 2, 1],
    [-34, -48, 2, 1],
    [-24, -71, 2, 1],
    [0, -7, 2, 1],
    [-9, -116, 2, 1],
    [-33, -38, 2, 1],
    [0, 0, 2, 1],
    [22, -32, 1, 1],
    [-145, -202, 2, 1],
    [142, -151, 2, 1],
    [0, 0, 2, 1],
    [0, -155, 2, 1],
    [-28, -41, 2, 1],
    [0, 0, 2, 1],
    [-46, -69, 1, 1],
    [-75, -79, 2, 1],
    [-28, -40, 2, 1],
    [306, -307, 2, 1],
    [0, 0, 2, 1],
    [10, -69, 2, 1],
    [-39, -45, 2, 1],
    [0, 0, 2, 1],
    [-37, -45, 2, 1],
    [74, -54, 2, 1],
    [74, 61, 2, 1],
    [46, -81, 2, 1],
    [-29, -40, 2, 1],
    [0, 0, 2, 1],
    [-37, -81, 1, 1],
    [-115, -119, 2, 1],
    [19, -66, 2, 1],
    [0, -20, 2, 1],
    [16, -53, 2, 1],
    [-27, -37, 2, 1],
    [7, 0, 2, 1],
    [-32, -128, 1, 1],
    [-87, -160, 1, 1],
    [-128, -166, 1, 1],
    [-140, -166, 1, 1],
    [-57, -67, 2, 1],
    [-26, -36, 2, 1],
    [0, -241, 2, 1],
    [5, -4, 2, 1],
    [182, -149, 2, 1],
    [-31, -35, 2, 1],
    [0, 0, 2, 1],
    [-44, -89, 1, 1],
    [-50, -89, 1, 1],
    [-55, -89, 1, 1],
    [-47, -182, 2, 1],
    [-34, -38, 2, 1],
    [0, -613, 2, 1],
    [64, 44, 2, 1],
    [0, -62, 2, 1],
    [-38, -40, 2, 1],
    [0, 0, 2, 1],
    [-90, -131, 1, 1],
    [-66, -89, 2, 1],
    [0, -117, 2, 1],
    [1, 0, 2, 1],
    [180, -165, 2, 1],
    [-38, -40, 2, 1],
    [0, 0, 2, 1],
    [0, 0, 1, 1],
    [-47, -63, 2, 1],
    [0, -145, 2, 1],
    [0, 0, 2, 1],
    [80, -80, 2, 1],
    [-30, -38, 2, 1],
    [16, 14, 2, 1],
    [-92, -119, 2, 1],
    [43, -64, 2, 1],
    [0, 0, 2, 1],
    [3, -367, 2, 1],
    [0, 0, 2, 1],
    [-27, -105, 1, 1],
    [-106, -205, 2, 1],
    [-34, -37, 2, 1],
    [58, -662, 2, 1],
    [23, 0, 2, 1],
    [58, -52, 2, 1],
    [-27, -48, 2, 1],
    [0, 0, 2, 1],
    [-13, -57, 1, 1],
    [-31, -70, 1, 1],
    [-57, -98, 1, 1],
    [-70, -104, 1, 1],
    [-98, -120, 1, 1],
    [-193, -203, 2, 1],
    [-11, -141, 2, 1],
    [-6, -11, 2, 1],
    [-5, -200, 2, 1],
    [-44, -49, 2, 1],
    [-6, -19, 2, 1],
    [-74, -81, 2, 1],
    [-18, -54, 2, 1],
    [57, -43, 2, 1],
    [-15, -57, 2, 1],
    [-30, -38, 2, 1],
    [39, 0, 2, 1],
    [-40, -127, 1, 1],
    [-94, -144, 1, 1],
    [-127, -149, 1, 1],
    [-90, -149, 1, 1],
    [-450, -454, 2, 1],
    [-26, -42, 2, 1],
    [32, -78, 2, 1],
    [-2, -6, 2, 1],
    [2, -87, 2, 1],
    [-41, -43, 2, 1],
    [0, 0, 2, 1],
    [-46, -61, 2, 1],
    [-26, -32, 2, 1],
    [81, -88, 2, 1],
    [0, -25, 2, 1],
    [-24, -57, 2, 1],
    [-31, -34, 2, 1],
    [0, 0, 2, 1],
    [-30, -93, 1, 1],
    [-69, -102, 1, 1],
    [-93, -102, 1, 1],
    [-47, -102, 1, 1],
    [-38, -61, 2, 1],
    [26, -69, 2, 1],
    [37, 10, 2, 1],
    [74, -76, 2, 1],
    [-37, -41, 2, 1],
    [0, 0, 2, 1],
    [-27, -61, 1, 1],
    [-42, -74, 2, 1],
    [61, -56, 2, 1],
    [7, 0, 2, 1],
    [37, -74, 2, 1],
    [0, 0, 2, 1],
    [-51, -96, 1, 1],
    [230, -87, 1, 1],
    [205, -87, 1, 1],
    [-39, -80, 2, 1],
    [12, -107, 2, 1],
    [29, -20, 2, 1],
    [51, -59, 2, 1],
    [0, 0, 2, 1],
    [0, -15, 1, 1],
    [-39, -61, 2, 1],
    [0, -71, 2, 1],
    [23, 11, 2, 1],
    [0, -83, 2, 1],
    [-30, -39, 2, 1],
    [0, 0, 2, 1],
    [-73, -82, 1, 1],
    [-63, -82, 1, 1],
    [-63, -68, 2, 1],
    [-34, -48, 2, 1],
    [78, -146, 2, 1],
    [-15, -22, 2, 1],
    [323, -289, 2, 1],
    [-40, -47, 2, 1],
    [0, 0, 2, 1],
    [-60, -94, 1, 1],
    [-81, -94, 1, 1],
    [-61, -94, 1, 1],
    [-71, -76, 2, 1],
    [-26, -31, 2, 1],
    [-12, -59, 2, 1],
    [3, -2, 2, 1],
    [89, -51, 2, 1],
    [-31, -45, 2, 1],
    [0, 0, 2, 1],
    [-35, -51, 1, 1],
    [-35, -57, 1, 1],
    [-46, -57, 1, 1],
    [-39, -57, 1, 1],
    [65, -42, 1, 1],
    [-55, -98, 2, 1],
    [-29, -45, 2, 1],
    [1, -135, 2, 1],
    [0, 0, 2, 1],
    [59, -130, 2, 1],
    [-35, -41, 2, 1],
    [11, 8, 2, 1],
    [0, -158, 1, 1],
    [-65, -218, 2, 1],
    [-26, -44, 2, 1],
    [-19, -85, 2, 1],
    [10, -32, 2, 1],
    [8, -128, 2, 1],
    [-31, -43, 2, 1],
    [3, -2, 2, 1],
    [-91, -111, 2, 1],
    [-39, -46, 2, 1],
    [49, -60, 2, 1],
    [36, 22, 2, 1],
    [-20, -111, 2, 1],
    [0, 0, 2, 1],
    [-156, -191, 2, 1],
    [-32, -48, 2, 1],
    [0, -51, 2, 1],
    [16, 0, 2, 1],
    [39, -72, 2, 1],
    [-32, -40, 2, 1],
    [0, 0, 2, 1],
    [-58, -81, 2, 1],
    [79, -75, 2, 1],
    [0, 0, 2, 1],
    [88, -134, 2, 1],
    [-36, -39, 2, 1],
    [29, 1, 2, 1],
    [-58, -72, 2, 1],
    [7, -126, 2, 1],
    [40, 0, 2, 1],
    [0, -96, 2, 1],
    [-38, -46, 2, 1],
    [0, 0, 2, 1],
    [-62, -66, 2, 1],
    [205, -167, 2, 1],
    [0, 0, 2, 1],
    [215, -188, 2, 1],
    [0, 0, 2, 1],
    [-63, -94, 1, 1],
    [-65, -106, 2, 1],
    [0, -63, 2, 1],
    [78, 55, 2, 1],
    [-3, -81, 2, 1],
    [-28, -37, 2, 1],
    [0, 0, 2, 1],
    [-40, -83, 1, 1],
    [-74, -84, 2, 1],
    [-34, -43, 2, 1],
    [0, -116, 2, 1],
    [0, 0, 2, 1],
    [84, -89, 2, 1],
    [-38, -41, 2, 1],
    [0, 0, 2, 1],
    [120, 100, 1, 1],
    [-161, -169, 2, 1],
    [-23, -76, 2, 1],
    [0, 0, 2, 1],
    [-8, -163, 2, 1],
    [0, 0, 2, 1],
    [0, -153, 1, 1],
    [-28, -52, 2, 1],
    [-26, -30, 2, 1],
    [11, -53, 2, 1],
    [0, -1, 2, 1],
    [0, -153, 2, 1],
    [-37, -40, 2, 1],
    [0, 0, 2, 1],
    [-15, -166, 1, 1],
    [220, -166, 1, 1],
    [-31, -61, 2, 1],
    [12, -105, 2, 1],
    [6, 2, 2, 1],
    [30, -221, 2, 1],
    [-38, -41, 2, 1],
    [0, 0, 2, 1],
    [-36, -52, 1, 1],
    [-75, -125, 1, 1],
    [-92, -125, 1, 1],
    [-84, -95, 2, 1],
    [0, -150, 2, 1],
    [22, -14, 2, 1],
    [39, -139, 2, 1],
    [0, 0, 2, 1],
    [-63, -74, 1, 1],
    [-33, -74, 2, 1],
    [116, -66, 2, 1],
    [7, -8, 2, 1],
    [268, -263, 2, 1],
    [0, 0, 2, 1],
    [-36, -56, 1, 1],
    [-36, -68, 1, 1],
    [-56, -78, 1, 1],
    [-68, -78, 1, 1],
    [-70, -78, 1, 1],
    [-84, -88, 2, 1],
    [1, -67, 2, 1],
    [0, 0, 2, 1],
    [-20, -65, 2, 1],
    [-37, -38, 2, 1],
    [0, 0, 2, 1],
    [1, 0, 1, 1],
    [-66, -89, 2, 1],
    [137, -163, 2, 1],
    [18, 0, 2, 1],
    [191, -160, 2, 1],
    [0, 0, 2, 1],
    [-42, -94, 1, 1],
    [-73, -105, 1, 1],
    [-42, -69, 2, 1],
    [0, -105, 2, 1],
    [0, -11, 2, 1],
    [-7, -94, 2, 1],
    [0, 0, 2, 1],
    [-43, -52, 1, 1],
    [-170, -373, 2, 1],
    [-26, -42, 2, 1],
    [26, -52, 2, 1],
    [5, -1, 2, 1],
    [85, -221, 2, 1],
    [-29, -35, 2, 1],
    [0, 0, 2, 1],
    [-62, -101, 1, 1],
    [-81, -101, 1, 1],
    [-43, -59, 2, 1],
    [-35, -38, 2, 1],
    [51, -89, 2, 1],
    [134, 126, 2, 1],
    [42, -61, 2, 1],
    [-33, -42, 2, 1],
    [0, 0, 2, 1],
    [-49, -93, 1, 1],
    [-65, -153, 2, 1],
    [58, -93, 2, 1],
    [31, 24, 2, 1],
    [0, -146, 2, 1],
    [0, 0, 2, 1],
    [-50, -69, 1, 1],
    [-59, -69, 1, 1],
    [-67, -136, 1, 1],
    [-52, -174, 2, 1],
    [-29, -33, 2, 1],
    [239, -239, 2, 1],
    [11, 3, 2, 1],
    [-7, -65, 2, 1],
    [-28, -31, 2, 1],
    [0, 0, 2, 1],
    [-39, -91, 1, 1],
    [-80, -91, 1, 1],
    [-69, -91, 1, 1],
    [-87, -122, 2, 1],
    [0, -80, 2, 1],
    [15, -11, 2, 1],
    [60, -107, 2, 1],
    [0, 0, 2, 1],
    [-32, -94, 1, 1],
    [-72, -94, 1, 1],
    [-92, -122, 1, 1],
    [-89, -103, 2, 1],
    [168, -143, 2, 1],
    [35, 19, 2, 1],
    [326, -187, 2, 1],
    [-34, -34, 2, 1],
    [21, 18, 2, 1],
    [-18, -28, 1, 1],
    [-132, -144, 2, 1],
    [79, -122, 2, 1],
    [33, 0, 2, 1],
    [-4, -309, 2, 1],
    [0, 0, 2, 1],
    [-38, -78, 1, 1],
    [-72, -78, 1, 1],
    [-68, -78, 1, 1],
    [-32, -76, 1, 1],
    [-48, -89, 2, 1],
    [46, -68, 2, 1],
    [20, 13, 2, 1],
    [63, -81, 2, 1],
    [-31, -42, 2, 1],
    [0, 0, 2, 1],
    [-33, -64, 1, 1],
    [-46, -64, 1, 1],
    [-55, -56, 2, 1],
    [-24, -185, 2, 1],
    [-5, -7, 2, 1],
    [251, -248, 2, 1],
    [-29, -36, 2, 1],
    [0, 0, 2, 1],
    [-39, -58, 1, 1],
    [-52, -58, 1, 1],
    [-61, -71, 2, 1],
    [-26, -41, 2, 1],
    [27, -70, 2, 1],
    [0, 0, 2, 1],
    [0, -453, 2, 1],
    [-27, -44, 2, 1],
    [0, 0, 2, 1]
])