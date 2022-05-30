%%% Images used in Hybrid Transfer Learning paper %%%

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

BinEdges = [400 451 502 553 604 655 706 757 808 859 910 961 1012 1063 1114 1165];
Fig3 = figure(3); b = histogram(data.cycle_life, 'BinWidth',...
    51, 'BinEdges', BinEdges, 'FaceAlpha', 1,'FaceColor', [0 0.4470 0.7410]); 
set(gcf,'Color',[1,1,1]);
xlabel('Cycle Life','fontsize',11,'fontweight','b');
ylabel('Number of Batteries','fontsize',11,'fontweight','b');
grid on; set(gca,'XMinorTick','on') 
set(gca,'FontSize',24,'Fontweight','b'); 

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
m = zeros(20,20);
for i = 1:height(data) % For each battery
    cap_dis = summary(i,:).QDischarge; % Discharging capacity of observed cell
    cap_chg = summary(i,:).QCharge; % Charging capacity
    ir = summary(i,:).IR; % Internal resistance
    tmax = summary(i,:).Tmax; tmin = summary(i,:).Tmin;
    tavg = summary(i,:).Tavg;
%     time = summary(i,:).chargetime;
    c = flip(summary(i,:).cycle); % fliping the cycle life to obtain the rul
    a = cell2mat(cycles(i));
    b = [];
    for j = cycle+1:length(a)
        % Using the difference between the variables from one cycle to
        % another to compare with the actual capacity value
%         cur_avg = mean(a(j).I) - mean(a(j-cycle+1).I);
%         cur_max = max(a(j).I) - max(a(j-cycle+1).I);
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
%         ir_increase = ir(j) - ir(j-cycle+1);
        
        mean_dQdV = mean(a(j).discharge_dQdV); % Use this one
        min_dQdV = min(a(j).discharge_dQdV);
        std_dQdV = std(a(j).discharge_dQdV); % Use this one
        
        mean_Qdlin = mean(a(j).Qdlin); % Use this one
        min_Qdlin = min(a(j).Qdlin);
        std_Qdlin = std(a(j).Qdlin); % Use this one
        
        mean_Tdlin = mean(a(j).Tdlin);
        min_Tdlin = min(a(j).Tdlin);
        std_Tdlin = std(a(j).Tdlin);
        
        b = [b; rul volt_time Qc_delta Qd_delta ...
            Qc Qd ir1 tavg(j) tmin(j) tmax(j) cur, ...
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

% Fig1 = figure(1); b = bar(corr, 'FaceColor', [0 0.4470 0.7410]); 
Fig1 = figure(1); b = bar(corr); 
set(gca, 'XTick', (1:1:length(corr)), 'XTickLabels', {'rul', 'volt_t', 'Qc_d',...
    'Qd_d', 'Qc', 'Qd', 'ir', 'tavg', 'tmin', 'tmax', 'curr',...
    'mean_dQdV', 'min_dQdV', 'var_dQdV', 'mean_Qdlin', 'min_Qdlin', 'var_Qdlin',...
    'mean_Tdlin', 'min_Tdlin', 'var_Tdlin'}, 'fontweight', 'b'); 
hold on; grid on; set(gcf, 'Color', [1 1 1]); 
set(gca, 'FontSize', 24); 
y = yline(0.6, '-r', 'Threshold', 'LineWidth', 2); 
y.LabelHorizontalAlignment = 'left'; y.FontSize = 18; 
ylabel("Pearson's correlation coefficient", 'fontweight', 'b');
title("Correlation analysis for different health indicators", 'fontweight', 'b');

Fig2 = figure(2); v = bar(var, 'FaceColor', [0 0.4470 0.7410]); 
set(gca, 'XTick', (1:1:length(var)), 'XTickLabels', {'rul', 'volt_t', 'Qc_d',...
    'Qd_d', 'Qc', 'Qd', 'ir', 'tavg', 'tmin', 'tmax', 'curr',...
    'mean_dQdV', 'min_dQdV', 'var_dQdV', 'mean_Qdlin', 'min_Qdlin', 'var_Qdlin',...
    'mean_Tdlin', 'min_Tdlin', 'var_Tdlin'}, 'fontweight', 'b'); 
hold on; grid on; set(gcf, 'Color', [1 1 1]); 
set(gca, 'FontSize', 24); 
y = yline(0.25, '-r', 'Threshold', 'LineWidth', 2); 
y.LabelHorizontalAlignment = 'left'; y.FontSize = 18; 
ylabel("Standard deviation of the correlation coefficient", 'fontweight', 'b');
title("Standard deviation analysis for the correlation of different health indicators", 'fontweight', 'b');

% Fig3 = figure(3);
% b = m([1 4 5 6 8 9 11 13 14 16 18 19 21], [1 4 5 6 8 9 11 13 14 16 18 19 21]);
% xvalues = {'rul', 'volt_t', 'Qc_d', 'Qd_d', 'Qd', 'ir',...
%     'tavg', 'tmax', 'curr', 'mean_dQdV', 'var_dQdV', 'mean_Qdlin', 'var_Qdlin'};
% yvalues = xvalues;
% h = heatmap(xvalues, yvalues, abs(b));
% title("Correlation between health indicators heatmap");
% set(gca, 'FontSize', 24); h.CellLabelFormat = '%.2f';

% Input image
Fig4 = figure(4);
i=1; a = cell2mat(data.cycles(i));
j=10; cycle=5; picRes=600;
recording_length = min(length(a(j).I), length(a(j+cycle-1).I));
I = [a(j).I(1:recording_length) a(j+cycle-1).I(1:recording_length)];
V = [a(j).V(1:recording_length) a(j+cycle-1).V(1:recording_length)];
Qc = [a(j).Qc(1:recording_length) a(j+cycle-1).Qc(1:recording_length)];
Qd = [a(j).Qd(1:recording_length) a(j+cycle-1).Qd(1:recording_length)];
subplot(2,2,1); hold on;
title('Current value over time');
plot(I(:,1), 'Color', 'r');
set(gca, 'YLim', [-7 7], 'XLim', [0 1000]);
xlabel('Time (s)'); ylabel('Current (A)');
subplot(2,2,2); hold on;
title('Charging capacity over time');
plot(Qc(:,1), 'Color', 'r');
set(gca, 'YLim', [0 1.2], 'XLim', [0 1000]);
xlabel('Time (s)'); ylabel('Capacity (Ah)');
subplot(2,2,3); hold on;
title('Discharging capacity over time');
plot(Qd(:,1), 'Color', 'r');
set(gca, 'YLim', [0 1.2], 'XLim', [0 1000]);
xlabel('Time (s)'); ylabel('Capacity (Ah)');
subplot(2,2,4); hold on;
title('Voltage curve over time');
plot(V(:,1), 'Color', 'r');
set(gca, 'YLim', [0 4], 'XLim', [0 1000]);
xlabel('Time (s)'); ylabel('Voltage (V)');
j=100;
recording_length = min(length(a(j).I), length(a(j+cycle-1).I));
I = [a(j).I(1:recording_length) a(j+cycle-1).I(1:recording_length)];
V = [a(j).V(1:recording_length) a(j+cycle-1).V(1:recording_length)];
Qc = [a(j).Qc(1:recording_length) a(j+cycle-1).Qc(1:recording_length)];
Qd = [a(j).Qd(1:recording_length) a(j+cycle-1).Qd(1:recording_length)];
subplot(2,2,1);
plot(I(:,1), 'Color', 'b');
set(gca, 'YLim', [-7 7], 'XLim', [0 1000]);
set(gca, 'FontSize', 24, 'GridLineStyle', ':', 'Color', [1 1 1]);
subplot(2,2,2);
plot(Qc(:,1), 'Color', 'b');
set(gca, 'YLim', [0 1.2], 'XLim', [0 1000]);
set(gca, 'FontSize', 24, 'GridLineStyle', ':', 'Color', [1 1 1]);
subplot(2,2,3);
plot(Qd(:,1), 'Color', 'b');
set(gca, 'YLim', [0 1.2], 'XLim', [0 1000]);
set(gca, 'FontSize', 24, 'GridLineStyle', ':', 'Color', [1 1 1]);
subplot(2,2,4);
plot(V(:,1), 'Color', 'b');
set(gca, 'YLim', [0 4], 'XLim', [0 1000]);
set(gca, 'FontSize', 24, 'GridLineStyle', ':', 'Color', [1 1 1]);

Fig5 = figure(5);
i=1; a = cell2mat(data.cycles(i));
j=10; cycle=10;
recording_length = min(length(a(j).I), length(a(j+cycle-1).I));
I = [a(j).I(1:recording_length) a(j+cycle-1).I(1:recording_length)];
V = [a(j).V(1:recording_length) a(j+cycle-1).V(1:recording_length)];
Qc = [a(j).Qc(1:recording_length) a(j+cycle-1).Qc(1:recording_length)];
Qd = [a(j).Qd(1:recording_length) a(j+cycle-1).Qd(1:recording_length)];
subplot(2,2,1, 'Position', [0 0.5 0.5 0.5]); hold on;
plot(I(:,1), 'Color', 'r', 'LineWidth', 1);
plot(I(:,2), 'Color', 'b', 'LineWidth', 1);
set(gca, 'YTickLabel', [], 'XTickLabel', [],...
    'YLim', [-7 7], 'XLim', [0 1000]);
subplot(2,2,2, 'Position', [0.5 0.5 0.5 0.5]); hold on;
plot(Qc(:,1), 'Color', 'r', 'LineWidth', 1);
plot(Qc(:,2), 'Color', 'b', 'LineWidth', 1);
set(gca, 'YTickLabel', [], 'XTickLabel', [],...
    'YLim', [0 1.2], 'XLim', [0 1000]);
subplot(2,2,3, 'Position', [0 0 0.5 0.5]); hold on;
plot(Qd(:,1), 'Color', 'r', 'LineWidth', 1);
plot(Qd(:,2), 'Color', 'b', 'LineWidth', 1);
set(gca, 'YTickLabel', [], 'XTickLabel', [],...
    'YLim', [0 1.2], 'XLim', [0 1000]);
subplot(2,2,4, 'Position', [0.5 0 0.5 0.5]); hold on;
plot(V(:,1), 'Color', 'r', 'LineWidth', 1);
plot(V(:,2), 'Color', 'b', 'LineWidth', 1);
set(gca, 'YTickLabel', [], 'XTickLabel', [],...
    'YLim', [0 4], 'XLim', [0 1000]);





















