%% Data visualization %%
clear; clc;
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
summary = data.summary;

%% Discharging Capacity
% We can see some significant outliers which can easily be removed. We also
% make use of the same loop to smooth out the curve
figure();
subplot(2,3,1); hold on; title('discharging capacity');
for i = 1:height(summary)
    a = summary(i,:).QDischarge;
    a = filloutliers(a, 'linear', 'movmedian', 5);
    a = smoothdata(a);
    summary(i,:).QDischarge = a;
    plot(a)
end

%% Charging Capacity
subplot(2,3,2); hold on; title('charging capacity');
for i = 1:height(summary)
    a = summary(i,:).QCharge;
    a = filloutliers(a, 'linear', 'movmedian', 5);
    a = smoothdata(a);
    summary(i,:).QCharge = a;
    plot(a)
end

% We can use both the charging and discharging capacities as inputs to the
% NN since they are perfectly correlated. (Perfect score of 1 using
% Pearson's correlation coefficient on each battery)

%% Internal Resistance
subplot(2,3,3); hold on; title('internal resistance')
for i = 1:height(summary)
    a = summary(i,:).IR;
    a = filloutliers(a, 'linear', 'movmedian', 10);
    a = smoothdata(a);
    summary(i,:).IR = a;
    plot(a)
end
    
%% Maximum Temperature (seems to be way too volatile to be useful)
subplot(2,3,4); hold on; title('maximal temperature')
for i = 1:height(summary)
    a = summary(i,:).Tmax;
    a = filloutliers(a, 'linear', 'movmedian', 50);
    a = smoothdata(a);
    summary(i,:).Tmax = a;
    plot(a)
end
    
%% Average Temperature (also too volatile)
subplot(2,3,5); hold on; title('average temperature')
for i = 1:height(summary)
    a = summary(i,:).Tavg;
    a = filloutliers(a, 'linear', 'movmedian', 50);
    a = smoothdata(a);
    summary(i,:).Tavg = a;
    plot(a)
end

%% Charging Time (doesn't seem to hold much useful information)
subplot(2,3,6); hold on; title('charging time')
for i = 1:height(summary)
    a = summary(i,:).chargetime;
    a = filloutliers(a, 'linear', 'movmedian', 50);
    a = smoothdata(a);
    summary(i,:).chargetime = a;
    plot(a)
end

%% Current Curves
% Each battery has a current curve for each cycle, giving us an enormous
% amount of data to play with
cycles = data.cycles;
figure();
subplot(2,3,1);
hold on; title('current')
for i = 1
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).I;
        b = filloutliers(b, 'linear', 'movmedian', 50);
        b = smooth(b);
        plot(b)
    end
end

%% Voltage Curves
subplot(2,3,2); 
hold on; title('voltage')
for i = 1
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).V;
        plot(b)
    end
end

%% Temperature Curves
subplot(2,3,3); hold on; title('temperature')
for i = 1
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).T;
        b = smooth(b);
        plot(b)
    end
end  

%% Capacity Curves
subplot(2,3,4); hold on; title('charging capacity over time')
for i = 1
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).Qc;
        plot(b)
    end
end  
subplot(2,3,5); hold on; title('discharging capacity over time')
for i = 1
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).Qd;
        plot(b)
    end
end

%% dQ/dV
subplot(2,3,6); hold on; title('Tdlin')
for i = 1
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).Tdlin;
        plot(b)
    end
end








