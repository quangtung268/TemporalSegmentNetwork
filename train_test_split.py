import os
import random

root_path = 'data/RGB/'
videos = os.listdir(root_path)

class_name_50 = os.listdir('C:/Users/Quangtung/PycharmProjects/Project2/UCF50/UCF50')

class_names = {}
for i, name in enumerate(class_name_50[:15]):
    class_names[name] = i

class_names['Billards'] = class_names.pop('Billiards')

written_file = []
for vid_name in videos:
    for class_name in class_names.keys():
        if class_name in vid_name:
            vid_path = os.path.join(root_path, vid_name)
            count = len(os.listdir(vid_path))
            row = vid_path + ' ' + str(count) + ' ' + str(class_names[class_name])
            written_file.append(row)

len = len(written_file)
random.shuffle(written_file)
train_len = len * 2 // 3

with open('data_split/train_ucf.txt', 'w') as f:
    for line in written_file[:train_len]:
        f.write(line)
        f.write('\n')

with open('data_split/test_ucf.txt', 'w') as f:
    for line in written_file[train_len:]:
        f.write(line)
        f.write('\n')
