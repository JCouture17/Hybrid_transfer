function [] = make_image(data, cycle, layout, imageFileName, rulFileName, hiFileName)
rul = [];
b = [];
summary = data.summary;

options.append = true;
options.color = true;
options.compress = 'lzw';
picRes = 224;

if cycle == 1
    for i = 1:height(data)
        a = cell2mat(data.cycles(i));
        c = flip(summary(i,:).cycle);
        ir = summary(i,:).IR;
        tmax = summary(i,:).Tmax;
        tavg = summary(i,:).Tavg;
        for j = cycle+1:length(a)-cycle
            close all
            I = a(j).I;
            V = a(j).V;
            Qc = a(j).Qc;
            Qd = a(j).Qd;
            rul = [rul; c(j+cycle)];
            
            figure('Position', [680 558 picRes picRes], 'visible', 'off');
            subplot('Position', [0 0.5 0.5 0.5]); hold on; 
            plot(I, 'Color', 'r'); 
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [-7 7], 'XLim', [0 1000]);
            subplot('Position', [0.5 0.5 0.5 0.5]); hold on; 
            plot(Qc, 'Color', 'r');  
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 1.2], 'XLim', [0 1000]);
            subplot('Position', [0 0 0.5 0.5]); hold on; 
            plot(Qd, 'Color', 'r'); 
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 1.2], 'XLim', [0 1000]);
            subplot('Position', [0.5 0 0.5 0.5]); hold on;
            plot(V, 'Color', 'r');
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 4], 'XLim', [0 1000]);

            saveastiff(getframe(gcf).cdata, imageFileName, options);
            
            %% Creating the health indicator dataset

            % Using the difference between the variables from one cycle to
            % another to compare with the actual capacity value
            % Time between 3.15 V and 3.3 V

            x = (a(j).V(find(a(j).V >= 3.3, 1)));
            volt_time = a(j).t(find((a(j).V == x),1,'first')) - ...
                a(j).t(find(a(j).V(find(a(j).V == x):end) <= 3.15,1));
            curr = mean(a(j).I); % Indicator of the cycling protocol used
%             Qc_delta = max(a(j).Qc); These are no longer useful
%             Qd_delta = max(a(j).Qd);
            Qd = max(a(j).Qd);
            rul_hi = c(j);
            ir1 = ir(j);

            mean_dQdV = mean(a(j).discharge_dQdV);
            std_dQdV = std(a(j).discharge_dQdV);
            mean_Qdlin = mean(a(j).Qdlin);
            std_Qdlin = std(a(j).Qdlin);

            b = [b; rul_hi volt_time curr  ...
                Qd ir1 tavg(j) tmax(j) mean_dQdV std_dQdV ...
                mean_Qdlin std_Qdlin];
        end
    end
    save(hiFileName, 'b');
elseif cycle ~= 1 && layout == 's'
    for i = 1:height(data) % For each battery
        a = cell2mat(data.cycles(i));
        c = flip(summary(i,:).cycle);
        ir = summary(i,:).IR; % Internal resistance
        tmax = summary(i,:).Tmax;
        tavg = summary(i,:).Tavg;
        for j = cycle+1:length(a)-cycle % For each cycle of each battery
            %% Making the image dataset
            close all
            recording_length = min(length(a(j).I), length(a(j+cycle-1).I));
            I = [a(j).I(1:recording_length) a(j+cycle-1).I(1:recording_length)];
            V = [a(j).V(1:recording_length) a(j+cycle-1).V(1:recording_length)];
            Qc = [a(j).Qc(1:recording_length) a(j+cycle-1).Qc(1:recording_length)];
            Qd = [a(j).Qd(1:recording_length) a(j+cycle-1).Qd(1:recording_length)];
            rul = [rul; c(j+cycle)];
            figure('Position', [680 558 picRes picRes], 'visible', 'off');
            subplot('Position', [0 0.5 0.5 0.5]); hold on; 
            plot(I(:,1), 'Color', 'r'); 
            plot(I(:,2), 'Color', 'b'); 
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [-7 7], 'XLim', [0 1000]);
            subplot('Position', [0.5 0.5 0.5 0.5]); hold on; 
            plot(Qc(:,1), 'Color', 'r'); 
            plot(Qc(:,2), 'Color', 'b'); 
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 1.2], 'XLim', [0 1000]);
            subplot('Position', [0 0 0.5 0.5]); hold on; 
            plot(Qd(:,1), 'Color', 'r'); 
            plot(Qd(:,2), 'Color', 'b'); 
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 1.2], 'XLim', [0 1000]);
            subplot('Position', [0.5 0 0.5 0.5]); hold on;
            plot(V(:,1), 'Color', 'r');
            plot(V(:,2), 'Color', 'b');
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 4], 'XLim', [0 1000]);

            saveastiff(getframe(gcf).cdata, imageFileName, options);

            %% Creating the health indicator dataset

            % Using the difference between the variables from one cycle to
            % another to compare with the actual capacity value
            % Time between 3.15 V and 3.3 V

            x = (a(j+cycle-1).V(find(a(j+cycle-1).V >= 3.3, 1)));
            volt_time = a(j+cycle-1).t(find((a(j+cycle-1).V == x),1,'first')) - ...
                a(j+cycle-1).t(find(a(j+cycle-1).V(find(a(j+cycle-1).V == x):end) <= 3.15,1));
            curr = mean(a(j+cycle-1).I); % Indicator of the cycling protocol used
            Qc_delta = max(a(j+cycle-1).Qc) - max(a(j).Qc);
            Qd_delta = max(a(j+cycle-1).Qd) - max(a(j).Qd);
            Qd = max(a(j+cycle-1).Qd);
            rul_hi = c(j+cycle-1);
            ir1 = ir(j+cycle-1);

            mean_dQdV = mean(a(j).discharge_dQdV);
            std_dQdV = std(a(j).discharge_dQdV);
            mean_Qdlin = mean(a(j).Qdlin);
            std_Qdlin = std(a(j).Qdlin);

            b = [b; rul_hi volt_time curr Qc_delta Qd_delta ...
                Qd ir1 tavg(j) tmax(j) mean_dQdV std_dQdV ...
                mean_Qdlin std_Qdlin];
        end
    end
    save(hiFileName, 'b');
elseif cycle ~= 1 && layout == 'm'
    for i = 1:height(data) % For each battery
        a = cell2mat(data.cycles(i));
        c = flip(summary(i,:).cycle);
        ir = summary(i,:).IR; % Internal resistance
        tmax = summary(i,:).Tmax;
        tavg = summary(i,:).Tavg;
        for j = cycle+1:length(a)-cycle % For each cycle of each battery
            %% Making the image dataset
            close all
            RL = [];
            for m = j:j+cycle-1
                RL = [RL length(a(m).I)];
            end
            recording_length = min(RL);
            I = []; V = []; Qc = []; Qd = [];
            for m = j:j+cycle-1
                I = [I a(m).I(1:recording_length)];
                V = [V a(m).V(1:recording_length)];
                Qc = [Qc a(m).Qc(1:recording_length)];
                Qd = [Qd a(m).Qd(1:recording_length)];
            end
            
            rul = [rul; c(j+cycle)];
            figure('Position', [680 558 picRes picRes], 'visible', 'off');
            subplot('Position', [0 0.5 0.5 0.5]); hold on; 
            plot(I(:,1:cycle));  
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [-7 7], 'XLim', [0 1000]);
            subplot('Position', [0.5 0.5 0.5 0.5]); hold on; 
            plot(Qc(:,1:cycle)); 
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 1.2], 'XLim', [0 1000]);
            subplot('Position', [0 0 0.5 0.5]); hold on; 
            plot(Qd(:,1:cycle)); 
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 1.2], 'XLim', [0 1000]);
            subplot('Position', [0.5 0 0.5 0.5]); hold on;
            plot(V(:,1:cycle));
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 4], 'XLim', [0 1000]);

            saveastiff(getframe(gcf).cdata, imageFileName, options);

            %% Creating the health indicator dataset

            % Using the difference between the variables from one cycle to
            % another to compare with the actual capacity value
            % Time between 3.15 V and 3.3 V

            x = (a(j+cycle-1).V(find(a(j+cycle-1).V >= 3.3, 1)));
            volt_time = a(j+cycle-1).t(find((a(j+cycle-1).V == x),1,'first')) - ...
                a(j+cycle-1).t(find(a(j+cycle-1).V(find(a(j+cycle-1).V == x):end) <= 3.15,1));
            curr = mean(a(j+cycle-1).I); % Indicator of the cycling protocol used
            Qc_delta = max(a(j+cycle-1).Qc) - max(a(j).Qc);
            Qd_delta = max(a(j+cycle-1).Qd) - max(a(j).Qd);
            Qd = max(a(j+cycle-1).Qd);
            rul_hi = c(j+cycle-1);
            ir1 = ir(j+cycle-1);

            mean_dQdV = mean(a(j).discharge_dQdV);
            std_dQdV = std(a(j).discharge_dQdV);
            mean_Qdlin = mean(a(j).Qdlin);
            std_Qdlin = std(a(j).Qdlin);

            b = [b; rul_hi volt_time curr Qc_delta Qd_delta ...
                Qd ir1 tavg(j) tmax(j) mean_dQdV std_dQdV ...
                mean_Qdlin std_Qdlin];
        end
    end
    save(hiFileName, 'b');  
% elseif layout == 't'
%     for i = 1:height(data) % For each battery
%         a = cell2mat(data.cycles(i));
%         c = flip(summary(i,:).cycle);
%         ir = summary(i,:).IR; % Internal resistance
%         tmax = summary(i,:).Tmax;
%         tavg = summary(i,:).Tavg;
%         for j = cycle+1:length(a)-cycle % For each cycle of each battery
%             %% Making the image dataset
%             close all
%             recording_length = max(length(a(j).I), length(a(j+cycle-1).I));
%             x1 = a(j).I; x2 = a(j+cycle-1).I;
%             I = [interp1([1:length(x1)],x1,[1:recording_length], 'pchip', 'extrap')', ...
%                 interp1([1:length(x2)],x2,[1:recording_length], 'pchip', 'extrap')']; 
%             
%             x1 = a(j).V; x2 = a(j+cycle-1).V;
%             V = [interp1([1:length(x1)],x1,[1:recording_length], 'pchip', 'extrap')', ...
%                 interp1([1:length(x2)],x2,[1:recording_length], 'pchip', 'extrap')'];
% 
%             x1 = a(j).Qc; x2 = a(j+cycle-1).Qc;
%             Qc = [interp1([1:length(x1)],x1,[1:recording_length], 'pchip', 'extrap')', ...
%                 interp1([1:length(x2)],x2,[1:recording_length], 'pchip', 'extrap')'];
%             
%             x1 = a(j).Qd; x2 = a(j+cycle-1).Qd;
%             Qd = [interp1([1:length(x1)],x1,[1:recording_length], 'pchip', 'extrap')', ...
%                 interp1([1:length(x2)],x2,[1:recording_length], 'pchip', 'extrap')'];
% 
%             rul = [rul; c(j+cycle)];
%             figure('Position', [680 558 picRes picRes], 'visible', 'off');
%             subplot('Position', [0 0.5 0.5 0.5]); hold on; 
%             plot(I(:,1), 'Color', 'r'); 
%             plot(I(:,2), 'Color', 'b'); 
%             set(gca, 'YTickLabel', [], 'XTickLabel', [],...
%                 'YLim', [-7 7], 'XLim', [0 1000]);
%             subplot('Position', [0.5 0.5 0.5 0.5]); hold on; 
%             plot(Qc(:,1), 'Color', 'r'); 
%             plot(Qc(:,2), 'Color', 'b'); 
%             set(gca, 'YTickLabel', [], 'XTickLabel', [],...
%                 'YLim', [0 1.2], 'XLim', [0 1000]);
%             subplot('Position', [0 0 0.5 0.5]); hold on; 
%             plot(Qd(:,1), 'Color', 'r'); 
%             plot(Qd(:,2), 'Color', 'b'); 
%             set(gca, 'YTickLabel', [], 'XTickLabel', [],...
%                 'YLim', [0 1.2], 'XLim', [0 1000]);
%             subplot('Position', [0.5 0 0.5 0.5]); hold on;
%             plot(V(:,1), 'Color', 'r');
%             plot(V(:,2), 'Color', 'b');
%             set(gca, 'YTickLabel', [], 'XTickLabel', [],...
%                 'YLim', [0 4], 'XLim', [0 1000]);
% 
%             saveastiff(getframe(gcf).cdata, imageFileName, options);
% 
%             %% Creating the health indicator dataset
% 
%             % Using the difference between the variables from one cycle to
%             % another to compare with the actual capacity value
%             % Time between 3.15 V and 3.3 V
% 
%             x = (a(j+cycle-1).V(find(a(j+cycle-1).V >= 3.3, 1)));
%             volt_time = a(j+cycle-1).t(find((a(j+cycle-1).V == x),1,'first')) - ...
%                 a(j+cycle-1).t(find(a(j+cycle-1).V(find(a(j+cycle-1).V == x):end) <= 3.15,1));
%             curr = mean(a(j+cycle-1).I); % Indicator of the cycling protocol used
%             Qc_delta = max(a(j+cycle-1).Qc) - max(a(j).Qc);
%             Qd_delta = max(a(j+cycle-1).Qd) - max(a(j).Qd);
%             Qd = max(a(j+cycle-1).Qd);
%             rul_hi = c(j+cycle-1);
%             ir1 = ir(j+cycle-1);
% 
%             mean_dQdV = mean(a(j).discharge_dQdV);
%             std_dQdV = std(a(j).discharge_dQdV);
%             mean_Qdlin = mean(a(j).Qdlin);
%             std_Qdlin = std(a(j).Qdlin);
% 
%             b = [b; rul_hi volt_time curr Qc_delta Qd_delta ...
%                 Qd ir1 tavg(j) tmax(j) mean_dQdV std_dQdV ...
%                 mean_Qdlin std_Qdlin];
%         end
%     end
%     save(hiFileName, 'b');
elseif layout == 't'
    for i = 1:height(data) % For each battery
        a = cell2mat(data.cycles(i));
        c = flip(summary(i,:).cycle);
        ir = summary(i,:).IR; % Internal resistance
        tmax = summary(i,:).Tmax;
        tavg = summary(i,:).Tavg;
        for j = cycle+1:length(a)-cycle % For each cycle of each battery
            %% Making the image dataset
            close all
            rul = [rul; c(j+cycle)];
            
            figure('Position', [680 558 picRes picRes], 'visible', 'off');
            subplot('Position', [0 0.5 0.5 0.5]); hold on; 
            plot(a(j).I, 'Color', 'r'); 
            plot(a(j+cycle-1).I, 'Color', 'b'); 
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [-7 7], 'XLim', [0 1000]);
            subplot('Position', [0.5 0.5 0.5 0.5]); hold on; 
            plot(a(j).Qc, 'Color', 'r'); 
            plot(a(j+cycle-1).Qc, 'Color', 'b'); 
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 1.2], 'XLim', [0 1000]);
            subplot('Position', [0 0 0.5 0.5]); hold on; 
            plot(a(j).Qd, 'Color', 'r'); 
            plot(a(j+cycle-1).Qd, 'Color', 'b'); 
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 1.2], 'XLim', [0 1000]);
            subplot('Position', [0.5 0 0.5 0.5]); hold on;
            plot(a(j).V, 'Color', 'r');
            plot(a(j+cycle-1).V, 'Color', 'b');
            set(gca, 'YTickLabel', [], 'XTickLabel', [],...
                'YLim', [0 4], 'XLim', [0 1000]);

            saveastiff(getframe(gcf).cdata, imageFileName, options);

            %% Creating the health indicator dataset

            % Using the difference between the variables from one cycle to
            % another to compare with the actual capacity value
            % Time between 3.15 V and 3.3 V

            x = (a(j+cycle-1).V(find(a(j+cycle-1).V >= 3.3, 1)));
            volt_time = a(j+cycle-1).t(find((a(j+cycle-1).V == x),1,'first')) - ...
                a(j+cycle-1).t(find(a(j+cycle-1).V(find(a(j+cycle-1).V == x):end) <= 3.15,1));
            curr = mean(a(j+cycle-1).I); % Indicator of the cycling protocol used
            Qc_delta = max(a(j+cycle-1).Qc) - max(a(j).Qc);
            Qd_delta = max(a(j+cycle-1).Qd) - max(a(j).Qd);
            Qd = max(a(j+cycle-1).Qd);
            rul_hi = c(j+cycle-1);
            ir1 = ir(j+cycle-1);

            mean_dQdV = mean(a(j).discharge_dQdV);
            std_dQdV = std(a(j).discharge_dQdV);
            mean_Qdlin = mean(a(j).Qdlin);
            std_Qdlin = std(a(j).Qdlin);

            b = [b; rul_hi volt_time curr Qc_delta Qd_delta ...
                Qd ir1 tavg(j) tmax(j) mean_dQdV std_dQdV ...
                mean_Qdlin std_Qdlin];
        end
    end
    save(hiFileName, 'b');
end
end
