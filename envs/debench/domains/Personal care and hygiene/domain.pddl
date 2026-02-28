;; Bucket: Personal care and hygiene
(define (domain elderly-homecare-hygiene)
  (:requirements :strips :typing)
  (:types
    robot person
    room
    item storage
  )
  (:predicates
    (at ?x - (either robot person item storage) ?r - room)
    (connected ?from - room ?to - room)
    (holding ?bot - robot ?it - item)
    (handfree ?bot - robot)
    (in ?it - item ?st - storage)
    (opened ?st - storage)
    (is-hygiene-item ?it - item)
    (available-in-room ?it - item ?r - room)
  )

  (:action move
    :parameters (?bot - robot ?from - room ?to - room)
    :precondition (and (at ?bot ?from) (connected ?from ?to))
    :effect (and (not (at ?bot ?from)) (at ?bot ?to))
  )

  (:action open-storage
    :parameters (?bot - robot ?st - storage ?r - room)
    :precondition (and (at ?bot ?r) (at ?st ?r))
    :effect (opened ?st)
  )

  (:action take-from-storage
    :parameters (?bot - robot ?it - item ?st - storage ?r - room)
    :precondition (and (at ?bot ?r) (at ?st ?r) (opened ?st) (in ?it ?st) (is-hygiene-item ?it) (handfree ?bot))
    :effect (and (holding ?bot ?it) (not (in ?it ?st)) (not (handfree ?bot)))
  )

  (:action pickup-item
    :parameters (?bot - robot ?it - item ?r - room)
    :precondition (and (at ?bot ?r) (at ?it ?r) (is-hygiene-item ?it) (handfree ?bot))
    :effect (and (holding ?bot ?it) (not (at ?it ?r)) (not (handfree ?bot)))
  )

  (:action deliver-hygiene-item
    :parameters (?bot - robot ?it - item ?p - person ?r - room)
    :precondition (and (at ?bot ?r) (at ?p ?r) (holding ?bot ?it) (is-hygiene-item ?it))
    :effect (and (available-in-room ?it ?r) (at ?it ?r) (not (holding ?bot ?it)) (handfree ?bot))
  )

  (:action putdown-item
    :parameters (?bot - robot ?it - item ?r - room)
    :precondition (and (at ?bot ?r) (holding ?bot ?it))
    :effect (and (at ?it ?r) (not (holding ?bot ?it)) (handfree ?bot))
  )
)
