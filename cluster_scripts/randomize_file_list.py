

H5_FILE_LIST = "/users/ak1774/scratch/esport/death_prediction/all_h5_files.txt"

with open(H5_FILE_LIST) as f:
        h5_files = f.readlines()

h5_files = [x.strip() for x in h5_files]

from random import shuffle
shuffle(h5_files)


num_files = len(h5_files)
num_training = int(num_files * 0.8)
num_rest = num_files - num_training
num_test = int(num_rest * 0.5)

training_files = h5_files[:num_training]
test_files = h5_files[num_training:num_training+num_test]
validation_files = h5_files[num_training+num_test:]

with open('training_files.txt', 'w') as f:
    for item in training_files:
        f.write("%s\n" % item)

with open('test_files.txt', 'w') as f:
    for item in test_files:
        f.write("%s\n" % item)

with open('validation_files.txt', 'w') as f:
    for item in validation_files:
        f.write("%s\n" % item)

        