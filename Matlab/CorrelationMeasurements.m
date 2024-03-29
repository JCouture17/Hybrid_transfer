%% Correlation Measurements %%
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

%% Correlation Coefficient
% Need to bring all of the arrays into the same size to be able to use
% Peason's correlation coefficient
summary = data.summary;
cycles = data.cycles;
cycle = 5;
r = [];
m = zeros(24,24);
for i = 1:height(data) % For each battery
    cap_dis = summary(i,:).QDischarge; % Discharging capacity of observed cell
    cap_chg = summary(i,:).QCharge; % Charging capacity
    ir = summary(i,:).IR; % Internal resistance
    tmax = summary(i,:).Tmax; tmin = summary(i,:).Tmin;
    tavg = summary(i,:).Tavg;
    time = summary(i,:).chargetime;
    c = flip(summary(i,:).cycle); % fliping the cycle life to obtain the rul
    a = cell2mat(cycles(i));
    b = [];
    for j = cycle+1:length(a)
        % Using the difference between the variables from one cycle to
        % another to compare with the actual capacity value
        cur_avg = mean(a(j).I) - mean(a(j-cycle+1).I);
        cur_max = max(a(j).I) - max(a(j-cycle+1).I);
        cur = mean(a(j).I);
        % Time between 3.15 V and 3.3 V
        x = (a(j).V(find(a(j).V >= 3.3, 1)));
        volt_time = a(j).t(find((a(j).V == x),1,'first')) - ...
            a(j).t(find(a(j).V(find(a(j).V == x):end) <= 3.15,1));
        Qc_delta = max(a(j).Qc) - max(a(j-cycle+1).Qc);
        Qd_delta = max(a(j).Qd) - max(a(j-cycle+1).Qd);
        Qc = max(a(j).Qc);
        Qd = max(a(j).Qd);
        rul = c(j);
        ir1 = ir(j);
        ir_increase = ir(j) - ir(j-cycle+1);
        
        mean_dQdV = mean(a(j).discharge_dQdV); % Use this one
        min_dQdV = min(a(j).discharge_dQdV);
        std_dQdV = std(a(j).discharge_dQdV); % Use this one
        
        mean_Qdlin = mean(a(j).Qdlin); % Use this one
        min_Qdlin = min(a(j).Qdlin);
        std_Qdlin = std(a(j).Qdlin); % Use this one
        
        mean_Tdlin = mean(a(j).Tdlin);
        min_Tdlin = min(a(j).Tdlin);
        std_Tdlin = std(a(j).Tdlin);
        
        b = [b; rul cur_avg cur_max volt_time Qc_delta Qd_delta ...
            Qc Qd ir1 ir_increase tavg(j) tmin(j) tmax(j) cur, time(j)...
            mean_dQdV min_dQdV std_dQdV mean_Qdlin min_Qdlin std_Qdlin...
            mean_Tdlin min_Tdlin std_Tdlin];
    end
    c = corrcoef(b);
    r = [r; c(1,:)];
    m = m + c;
end

m = m/height(data);

corr = mean(abs(r));
var = std(abs(r));

figure; b = bar(corr); 
set(gca, 'XTick', (1:1:length(corr)), 'XTickLabels', {'rul', 'cur_a', 'cur_m', 'volt_t', 'Qc_d',...
    'Qd_d', 'Qc', 'Qd', 'ir', 'ir_d', 'tavg', 'tmin', 'tmax', 'curr', 'chargetime',...
    'mean_dQdV', 'min_dQdV', 'var_dQdV', 'mean_Qdlin', 'min_Qdlin', 'var_Qdlin',...
    'mean_Tdlin', 'min_Tdlin', 'var_Tdlin'}, 'fontweight', 'b'); 
hold on; grid on; set(gcf, 'Color', [1 1 1]); 
set(gca, 'FontSize', 11, 'GridLineStyle', ':'); 
y = yline(0.6, '-r', 'Threshold', 'LineWidth', 2); 
y.LabelHorizontalAlignment = 'left'; y.FontSize = 18; 
ylabel("Pearson's correlation coefficient", 'fontweight', 'b');
title("Correlation analysis for different health indicators", 'fontweight', 'b');

figure; v = bar(var); 
set(gca, 'XTick', (1:1:length(var)), 'XTickLabels', {'rul', 'cur_a', 'cur_m', 'volt_t', 'Qc_d', 'Qd_d', ...
    'Qc', 'Qd', 'ir', 'ir_d', 'tavg', 'tmin', 'tmax', 'curr', 'chargetime', ...
    'mean_dQdV', 'min_dQdV', 'var_dQdV', 'mean_Qdlin', 'min_Qdlin', 'var_Qdlin', ...
    'mean_Tdlin', 'min_Tdlin', 'var_Tdlin'}, 'fontweight', 'b'); 
hold on; grid on; set(gcf, 'Color', [1 1 1]); 
set(gca, 'FontSize', 11, 'GridLineStyle', ':'); 
y = yline(0.25, '-r', 'Threshold', 'LineWidth', 2); 
y.LabelHorizontalAlignment = 'left'; y.FontSize = 18; 
ylabel("Standard deviation of the correlation coefficient", 'fontweight', 'b');
title("Standard deviation analysis for the correlation of different health indicators", 'fontweight', 'b');

% xvalues = {'rul', 'cur_a', 'cur_m', 'volt_t', 'Qc_d', 'Qd_d', ...
%                 'Qc', 'Qd', 'ir', 'ir_d', 'tavg', 'tmin', 'tmax', 'cur'};
% yvalues = {'rul', 'cur_a', 'cur_m', 'volt_t', 'Qc_d', 'Qd_d', ...
%                 'Qc', 'Qd', 'ir', 'ir_d', 'tavg', 'tmin', 'tmax', 'cur'};
% hold off; figure; h = heatmap(xvalues, yvalues, abs(m)); 
% title("Correlation coefficient heatmap");
% set(gca, 'FontSize', 11); h.CellLabelFormat = '%.2f';

