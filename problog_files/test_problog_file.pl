
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

happensAt(v, p1, 1, walking). happensAt(v, p2, 1, walking). % false
happensAt(v, p1, 2, walking). happensAt(v, p2, 2, running). % true
happensAt(v, p1, 3, walking). happensAt(v, p2, 3, walking). % false
happensAt(v, p1, 4, walking). happensAt(v, p2, 4, walking). % true
happensAt(v, p1, 5, walking). happensAt(v, p2, 5, running). % true

query(holdsAt(v, interacting(p1, p2), 1)).
query(holdsAt(v, interacting(p1, p2), 2)).
query(holdsAt(v, interacting(p1, p2), 3)).
query(holdsAt(v, interacting(p1, p2), 4)).
query(holdsAt(v, interacting(p1, p2), 5)).