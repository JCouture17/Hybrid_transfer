%% Health Indicator generation
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

%% Health indicators
% Qd; Time between voltages; Qd_d; Qc_d; IR; Tavg; Tmax
summary = data.summary;
cycles = data.cycles;
cycle = 5;
r = [];
b = [];
for i = 1:height(data) % For each battery
    cap_dis = summary(i,:).QDischarge; % Discharging capacity of observed cell
    cap_chg = summary(i,:).QCharge; % Charging capacity
    ir = summary(i,:).IR; % Internal resistance
    tmax = summary(i,:).Tmax; tmin = summary(i,:).Tmin;
    tavg = summary(i,:).Tavg;
    c = flip(summary(i,:).cycle); % fliping the cycle life to obtain the rul
    a = cell2mat(cycles(i));
    for j = cycle+1:length(a)
        % Using the difference between the variables from one cycle to
        % another to compare with the actual capacity value
        % Time between 3.15 V and 3.3 V
        x = (a(j).V(find(a(j).V >= 3.3, 1)));
        volt_time = a(j).t(find((a(j).V == x),1,'first')) - ...
            a(j).t(find(a(j).V(find(a(j).V == x):end) <= 3.15,1));
        curr = mean(a(j).I);
        Qc_delta = max(a(j).Qc) - max(a(j-cycle+1).Qc);
        Qd_delta = max(a(j).Qd) - max(a(j-cycle+1).Qd);
        Qd = max(a(j).Qd);
        rul = c(j);
        ir1 = ir(j);
        b = [b; rul volt_time curr Qc_delta Qd_delta ...
            Qd ir1 tavg(j) tmax(j)];
    end
end










