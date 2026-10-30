function grid = knn_grid(cfg)
%KNN_GRID The 120 tkNN settings: distance x k x distance weighting (Eq. 23).
grid = struct('distance', {}, 'k', {}, 'weight', {});
for d = 1:numel(cfg.distances)
    for k = cfg.kValues
        for w = 1:numel(cfg.weights)
            grid(end+1) = struct('distance', cfg.distances{d}, 'k', k, ...
                                 'weight', cfg.weights{w});            %#ok<AGROW>
        end
    end
end
end
