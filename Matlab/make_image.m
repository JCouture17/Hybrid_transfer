function [] = make_image(data, cycle, folder, rulFileName)
rul = [];
summary = data.summary;

options.append = true;
options.color = true;
options.compress = 'lzw';
picRes = 224;

for i = 1:height(data) % For each battery
    a = cell2mat(data.cycles(i));
    c = flip(summary(i,:).cycle);
    for j = cycle+1:length(a)-cycle % For each cycle of each battery
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

        saveastiff(getframe(gcf).cdata, folder, options);
    end
end

save(rulFileName, 'rul');
end

