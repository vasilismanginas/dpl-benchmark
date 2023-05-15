
nn(lstm_net, [X, P, T], Y, [walking, active, inactive, running]) :: simple_event(X, P, T, Y).

happens(X, T) :- simple_event(X, p1, T, active), simple_event(X, p2, T, active).



nn(lstm_net, [Video, P, T], SE, [active, inactive, walking, running]) :: happensAt(Video, P, T, SE).

% next(T1, T2) :- T2 = T1 + 1.

% maybe for the simple case we don't check if F is a fluent or
% what it means to be a fluent, and so the following is sufficient
% this translates to fluent F holds at time T2 of video V if ...

holdsAt(Video, F, T2) :- initiatedAt(F, Video, T2-1), T2>=1.
holdsAt(Video, F, T2) :- holdsAt(Video, F, T2-1), \+ terminatedAt(F, Video, T2), T2>=1.

initiatedAt(interacting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, walking),
    happensAt(Video, P2, T, walking).

terminatedAt(interacting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, walking),
    happensAt(Video, P2, T, running).

terminatedAt(interacting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, running),
    happensAt(Video, P2, T, walking).



