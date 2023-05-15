
% for the simple case we don't check:
%   if F is a fluent
%   the connection between people and events
%   the connection between time and events
% see activity_detection.pl for an ASP program containing the above (Andreas)
% for us the query is holdsAt(V, F, T) which translates to fluent F holds at time T of video V if ...


nn(lstm_net, [Video, P, T], SE, [active, inactive, walking, running]) :: happensAt(Video, P, T, SE).


holdsAt(Video, F, T) :- previous(T1, T), initiatedAt(F, Video, T1).
holdsAt(Video, F, T) :- previous(T1, T), holdsAt(Video, F, T1), \+terminatedAt(F, Video, T1).

initiatedAt(interacting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, walking),
    happensAt(Video, P2, T, walking).

terminatedAt(interacting(P1, P2), Video, T) :- 
    happensAt(Video, P1, T, walking),
    happensAt(Video, P2, T, running).

previous(T1, T) :- 
    T >= 0, 
    T1 is T-1, 
    T1 >= 0.