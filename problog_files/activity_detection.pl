
nn(lstm_net, [X, P, T], Y, [active, inactive, walking, running]) :: happensAt(X, P, T, Y)

% event calculus for the law of inertia: an event will continue to take place if it is 
% initiated by an action at a given time until it is interrupted by another action.
holdsAt(F, T2) :- initiatedAt(F, T1), fluent(F), next(T1, T2).
holdsAt(F, T2) :- holdsAt(F, T1), not terminatedAt(F, T1), fluent(F), next(T1, T2).

% define the interacting complex event as a fluent
fluent(interacting(P1, P2)) :- person(P1), person(P2), P1 != P2.

time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords( , , ,T).
time(T) :- orientation( , ,T).

person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).

initiatedAt(interacting(X0,X1),X2) :- 
    happensAt(active(X0),X2),
    happensAt(active(X1),X2),
    close(X0,X1,D,X2).

initiatedAt(interacting(X0,X1),X2) :- 
    happensAt(active(X0),X2),
    happensAt(inactive(X1),X2),
    close(X0,X1,D,X2).
    
initiatedAt(interacting(X0,X1),X2) :- 
    happensAt(inactive(X0),X2),
    happensAt(active(X1),X2),
    close(X0,X1,D,X2).

initiatedAt(interacting(X0,X1),X2) :- 
    happensAt(inactive(X0),X2),
    happensAt(inactive(X1),X2),
    close(X0,X1,D,X2).

terminatedAt(interacting(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(interacting(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(interacting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(interacting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(interacting(X0,X1),X2) :- happensAt(walking(X0),X2), far(X0,X1,D,X2).
terminatedAt(interacting(X0,X1),X2) :- happensAt(walking(X1),X2), far(X0,X1,D,X2).