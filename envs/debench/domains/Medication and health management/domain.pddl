;; Bucket: Medication and health management
(define (domain elderly-homecare-medication)
  (:requirements :strips :typing)
  (:types
    robot person
    room
    item container
  )
  (:predicates
    (at ?x - (either robot person item container) ?r - room)
    (connected ?from - room ?to - room)
    (holding ?bot - robot ?it - item)
    (handfree ?bot - robot)
    (in ?it - item ?c - container)
    (opened ?c - container)
    (is-pill ?it - item)
    (delivered ?it - item ?p - person)
    (meds-delivered ?p - person)
  )

  (:action move
    :parameters (?bot - robot ?from - room ?to - room)
    :precondition (and (at ?bot ?from) (connected ?from ?to))
    :effect (and (not (at ?bot ?from)) (at ?bot ?to))
  )

  (:action open-container
    :parameters (?bot - robot ?c - container ?r - room)
    :precondition (and (at ?bot ?r) (at ?c ?r))
    :effect (opened ?c)
  )

  (:action take-from-container
    :parameters (?bot - robot ?pill - item ?c - container ?r - room)
    :precondition (and (at ?bot ?r) (at ?c ?r) (opened ?c) (in ?pill ?c) (is-pill ?pill) (handfree ?bot))
    :effect (and (holding ?bot ?pill) (not (in ?pill ?c)) (not (handfree ?bot)))
  )

  (:action pickup-item
    :parameters (?bot - robot ?it - item ?r - room)
    :precondition (and (at ?bot ?r) (at ?it ?r) (handfree ?bot))
    :effect (and (holding ?bot ?it) (not (at ?it ?r)) (not (handfree ?bot)))
  )

  (:action deliver-item-to-person
    :parameters (?bot - robot ?it - item ?p - person ?r - room)
    :precondition (and (at ?bot ?r) (at ?p ?r) (holding ?bot ?it))
    :effect (and (delivered ?it ?p) (meds-delivered ?p) (at ?it ?r) (not (holding ?bot ?it)) (handfree ?bot))
  )
)
