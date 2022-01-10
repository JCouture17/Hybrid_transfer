%% Combine datasets
clear; clc;

path = '.\Data\';

%% Training images
t1 = loadtiff(append(path, 'train_data1.tif'));
t2 = loadtiff(append(path, 'train_data2.tif'));
t3 = loadtiff(append(path, 'train_data3.tif'));
t4 = loadtiff(append(path, 'train_data4.tif'));
t5 = loadtiff(append(path, 'train_data5.tif'));
t6 = loadtiff(append(path, 'train_data6.tif'));
t7 = loadtiff(append(path, 'train_data7.tif'));
t8 = loadtiff(append(path, 'train_data8.tif'));
r1 = load(append(path, 'train_rul1.mat')).rul;
r2 = load(append(path, 'train_rul2.mat')).rul;
r3 = load(append(path, 'train_rul3.mat')).rul;
r4 = load(append(path, 'train_rul4.mat')).rul;
r5 = load(append(path, 'train_rul5.mat')).rul;
r6 = load(append(path, 'train_rul6.mat')).rul;
r7 = load(append(path, 'train_rul7.mat')).rul;
r8 = load(append(path, 'train_rul8.mat')).rul;

training_dataset = cat(4, t1, t2, t3, t4, t5, t6, t7, t8);
training_targets = cat(1, r1, r2, r3, r4, r5, r6, r7, r8);

options.append = true;
options.color = true;
options.compress = 'lzw';
saveastiff(training_dataset, append(path, 'training_data.tif'), options);
save(append(path, 'training_targets.mat'), 'training_targets');

%% Testing images
t1 = loadtiff(append(path, 'test_data1.tif'));
t2 = loadtiff(append(path, 'test_data2.tif'));
r1 = load(append(path, 'test_rul1.mat')).rul;
r2 = load(append(path, 'test_rul2.mat')).rul;

testing_dataset = cat(4, t1, t2);
testing_targets = cat(1, r1, r2);

saveastiff(testing_dataset, append(path, 'testing_dataset.tif'), options);
save(append(path, 'testing_targets.mat'), 'testing_targets');


%% Training and Testing Health Indicators
t1 = load(append(path, 'train_hi1.mat')).b;
t2 = load(append(path, 'train_hi2.mat')).b;
t3 = load(append(path, 'train_hi3.mat')).b;
t4 = load(append(path, 'train_hi4.mat')).b;
t5 = load(append(path, 'train_hi5.mat')).b;
t6 = load(append(path, 'train_hi6.mat')).b;
t7 = load(append(path, 'train_hi7.mat')).b;
t8 = load(append(path, 'train_hi8.mat')).b;
r1 = load(append(path, 'test_hi1.mat')).b;
r2 = load(append(path, 'test_hi2.mat')).b;

training_his = cat(1, t1, t2, t3, t4, t5, t6, t7, t8);
testing_his = cat(1, r1, r2);

save(append(path, 'training_his.mat'), 'training_his');
save(append(path, 'testing_his.mat'), 'testing_his');














