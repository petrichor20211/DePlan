;; Bucket: Safety and emergency response
(define (domain elderly-homecare-safety)
  (:requirements :strips :typing)
  (:types
    robot person
    room
    item device
  )
  (:predicates
    (at ?x - (either robot person item device) ?r - room)
    (connected ?from - room ?to - room)
    (holding ?bot - robot ?it - (either item device))
    (handfree ?bot - robot)
    (light-on ?r - room)
    (dark ?r - room)
    (urgent ?p - person)
    (is-phone ?d - device)
    (help-requested ?p - person)
    (phone-with-person ?p - person)
    (assisted ?p - person)
    (bleeding-checked ?p - person)
  )

  (:action move
    :parameters (?bot - robot ?from - room ?to - room)
    :precondition (and (at ?bot ?from) (connected ?from ?to))
    :effect (and (not (at ?bot ?from)) (at ?bot ?to))
  )

  (:action pickup-thing
    :parameters (?bot - robot ?x - (either item device) ?r - room)
    :precondition (and (at ?bot ?r) (at ?x ?r) (handfree ?bot))
    :effect (and (holding ?bot ?x) (not (at ?x ?r)) (not (handfree ?bot)))
  )

  (:action deliver-phone
    :parameters (?bot - robot ?ph - device ?p - person ?r - room)
    :precondition (and (at ?bot ?r) (at ?p ?r) (holding ?bot ?ph) (is-phone ?ph))
    :effect (and (phone-with-person ?p) (at ?ph ?r) (not (holding ?bot ?ph)) (handfree ?bot))
  )

  (:action call-for-help
    :parameters (?bot - robot ?p - person ?r - room)
    :precondition (and (at ?bot ?r) (at ?p ?r) (urgent ?p) (phone-with-person ?p))
    :effect (help-requested ?p)
  )

  (:action switch-on-light
    :parameters (?bot - robot ?r - room)
    :precondition (and (at ?bot ?r) (dark ?r))
    :effect (and (light-on ?r) (not (dark ?r)))
  )

  (:action switch-off-light
    :parameters (?bot - robot ?r - room)
    :precondition (and (at ?bot ?r) (light-on ?r))
    :effect (and (dark ?r) (not (light-on ?r)))
  )

  (:action assist-to-seat
    :parameters (?bot - robot ?p - person ?r - room)
    :precondition (and (at ?bot ?r) (at ?p ?r))
    :effect (assisted ?p)
  )

  (:action check-bleeding
    :parameters (?bot - robot ?p - person ?r - room)
    :precondition (and (at ?bot ?r) (at ?p ?r))
    :effect (bleeding-checked ?p)
  )
)
