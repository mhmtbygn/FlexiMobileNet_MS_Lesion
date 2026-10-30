function p = apply_selection(sel, P)
%APPLY_SELECTION Apply a selection from IMV_SELECT to test predictions P (n x m).
if strcmp(sel.type, 'single')
    p = P(:, sel.members);
else
    p = mode(P(:, sel.members), 2);
end
end
