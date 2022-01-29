%% Init
clear; clc; close all;
run('LoadData.m');
data = struct2table(batch_combined);
cl = data.cycle_life;  
    % We're treating batteries with cycle life >1175 and <400 as outliers
    % We're losing 11 batteries to outlying tendencies
for i = 1:length(cl)
    if data.cycle_life(i) < 400 || data.cycle_life(i) > 1175
        data.cycle_life(i) = 0;
    end    
end
data = data(data.cycle_life ~= 0, :);

data.cycle_life([37:39, 42:48, 50, 53:56, 59]) = 0;
data = data(data.cycle_life ~= 0,:);
summary = data.summary;

%% Curve Smoothing
% Discharging Capacity
for i = 1:height(summary)
    b = summary(i,:).QDischarge;
    b = filloutliers(b, 'linear', 'movmedian', 5);
    b = smoothdata(b);
    summary(i,:).QDischarge = b;
end

% Charging Capacity
for i = 1:height(summary)
    b = summary(i,:).QCharge;
    b = filloutliers(b, 'linear', 'movmedian', 5);
    b = smoothdata(b);
    summary(i,:).QCharge = b;
end

% Current Curves
cycles = data.cycles;
for i = 1:length(cycles)
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).I;
        b = filloutliers(b, 'linear', 'movmedian', 50);
        b = smooth(b);
        a(j).I = b;
    end
    cycles(i) = {a};
end

% Voltage Curves
for i = 1:length(cycles)
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).V;
        b = smooth(b);
        a(j).V = b;
    end
    cycles(i) = {a};
end

data.cycles = cycles;

%% Train / Test Split
test = data(1:5:end,:);
data(1:5:end,:) = [];

%% Image as Inputs
folder = 'C:\Users\JCout\Documents\GitHub\Hybrid_transfer\Data\';
cycle_count = 10;
layout = 'm'; % 's' for 'single', 'm' for 'multi'

make_image(data(1:10,:), cycle_count, layout, append(folder, 'train_data1.tif'), append(folder, 'train_rul1.mat'),...
    append(folder, 'train_hi1.mat'));
make_image(data(11:20,:), cycle_count, layout, append(folder, 'train_data2.tif'), append(folder, 'train_rul2.mat'),...
    append(folder, 'train_hi2.mat'));
make_image(data(21:30,:), cycle_count, layout, append(folder, 'train_data3.tif'), append(folder, 'train_rul3.mat'),...
    append(folder, 'train_hi3.mat'));
make_image(data(31:40,:), cycle_count, layout, append(folder, 'train_data4.tif'), append(folder, 'train_rul4.mat'),...
    append(folder, 'train_hi4.mat'));
make_image(data(41:50,:), cycle_count, layout, append(folder, 'train_data5.tif'), append(folder, 'train_rul5.mat'),...
    append(folder, 'train_hi5.mat'));
make_image(data(51:60,:), cycle_count, layout, append(folder, 'train_data6.tif'), append(folder, 'train_rul6.mat'),...
    append(folder, 'train_hi6.mat'));
make_image(data(61:70,:), cycle_count, layout, append(folder, 'train_data7.tif'), append(folder, 'train_rul7.mat'),...
    append(folder, 'train_hi7.mat'));
make_image(data(71:height(data),:), cycle_count, layout, append(folder, 'train_data8.tif'), append(folder, 'train_rul8.mat'),...
    append(folder, 'train_hi8.mat'));

make_image(test(1:10,:), cycle_count, layout, append(folder, 'test_data1.tif'), append(folder, 'test_rul1.mat'),...
    append(folder, 'test_hi1.mat'));
make_image(test(10:height(test),:), cycle_count, layout, append(folder, 'test_data2.tif'), append(folder, 'test_rul2.mat'),...
    append(folder, 'test_hi2.mat'));

