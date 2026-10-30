function T = define_task(label, task)
%DEFINE_TASK Rows, labels and reference classifiers of each analysis.
%   label: 1 active T1, 2 active FLAIR, 3 chronic T1, 4 chronic FLAIR.
%   T.baseline is the all-chronic reference classifier on the same rows
%   (sequence-aware for the four-class task).
label = label(:);
switch task
    case 'fourclass'
        T.idx = (1:numel(label))';
        T.y = label;
        T.names = {'Active T1', 'Active FLAIR', 'Chronic T1', 'Chronic FLAIR'};
        T.baseline = label;
        T.baseline(label == 1) = 3;
        T.baseline(label == 2) = 4;
    case 'case1'                                 % sequence-recognition control
        T.idx = find(label <= 2);
        T.y = label(T.idx);
        T.names = {'Active T1', 'Active FLAIR'};
        T.baseline = [];
    case 'case2'                                 % active vs chronic, both sequences
        T.idx = (1:numel(label))';
        T.y = 1 + (label >= 3);
        T.names = {'Active', 'Chronic'};
        T.baseline = 2 * ones(numel(T.idx), 1);
    case 'flair_only'
        T.idx = find(label == 2 | label == 4);
        T.y = 1 + (label(T.idx) == 4);
        T.names = {'Active', 'Chronic'};
        T.baseline = 2 * ones(numel(T.idx), 1);
    case 't1_only'
        T.idx = find(label == 1 | label == 3);
        T.y = 1 + (label(T.idx) == 3);
        T.names = {'Active', 'Chronic'};
        T.baseline = 2 * ones(numel(T.idx), 1);
    otherwise
        error('define_task:unknown', 'Unknown task %s.', task);
end
T.task = task;
end
