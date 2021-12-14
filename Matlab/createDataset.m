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
folder = 'C:\Users\JCout\Documents\GitHub\Hybrid_resnet\Data\';

make_image(data(1:10,:), 3, append(folder, 'train_data1.tif'), append(folder, 'train_rul1.mat'));
make_image(data(11:20,:), 3, append(folder, 'train_data2.tif'), append(folder, 'train_rul2.mat'));
make_image(data(21:30,:), 3, append(folder, 'train_data3.tif'), append(folder, 'train_rul3.mat'));
make_image(data(31:40,:), 3, append(folder, 'train_data4.tif'), append(folder, 'train_rul4.mat'));
make_image(data(41:50,:), 3, append(folder, 'train_data5.tif'), append(folder, 'train_rul5.mat'));
make_image(data(51:60,:), 3, append(folder, 'train_data6.tif'), append(folder, 'train_rul6.mat'));
make_image(data(61:70,:), 3, append(folder, 'train_data7.tif'), append(folder, 'train_rul7.mat'));
make_image(data(71:height(data),:), 3, append(folder, 'train_data8.tif'), append(folder, 'train_rul8.mat'));

make_image(test(1:10,:), 3, append(folder, 'test_data1.tif'), append(folder, 'test_rul1.mat'));
make_image(test(10:height(test),:), 3, append(folder, 'test_data2.tif'), append(folder, 'test_rul2.mat'));

% %% 1 Cycle
% cycle = 1;
% for i = 1:height(data)
%     a = cell2mat(data.cycles(i));
%     c = flip(summary(i,:).cycle);
%     for j = cycle+1:length(a)-cycle
%         recording_length = min(length(a(j).I), length(a(j+cycle-1).I));
%         curr = a(j).I(1:recording_length);
%         V = a(j).V(1:recording_length);
%         Qc = a(j).Qc(1:recording_length);
%         Qd = a(j).Qd(1:recording_length);

%         rul = [rul; c(j+cycle)];
%         
%         fig = figure('Position', [680 558 picRes picRes], 'visible', 'off');
%         subplot(2,2,1, 'Position', [0 0.5 0.5 0.5]); hold on; 
%         plot(curr(:,1), 'Color', 'r');  
%         set(gca, 'YTickLabel', [], 'XTickLabel', [],...
%             'YLim', [-7 7], 'XLim', [0 1000]);
%         subplot(2,2,2, 'Position', [0.5 0.5 0.5 0.5]); hold on; 
%         plot(Qc(:,1), 'Color', 'r');  
%         set(gca, 'YTickLabel', [], 'XTickLabel', [],...
%             'YLim', [0 1.2], 'XLim', [0 1000]);
%         subplot(2,2,3, 'Position', [0 0 0.5 0.5]); hold on; 
%         plot(Qd(:,1), 'Color', 'r'); 
%         set(gca, 'YTickLabel', [], 'XTickLabel', [],...
%             'YLim', [0 1.2], 'XLim', [0 1000]);
%         subplot(2,2,4, 'Position', [0.5 0 0.5 0.5]); 

%         
%         f = getframe(gcf); new = f.cdata;
%         imwrite(new, folder, 'WriteMode', 'append');
%         end
% end
% save(rulFileName, 'rul');
