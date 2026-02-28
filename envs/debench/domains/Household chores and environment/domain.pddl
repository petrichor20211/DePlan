;; Bucket: Household chores and environment
(define (domain elderly-homecare-household)
  (:requirements :strips :typing)
  (:types
    robot person
    room
    item surface tool
  )
  (:predicates
    (at ?x - (either robot person item tool) ?r - room)
    (connected ?from - room ?to - room)
    (holding ?bot - robot ?it - (either item tool))
    (handfree ?bot - robot)
    (on ?it - item ?s - surface)
    (surface-in ?s - surface ?r - room)
    (tool-for-cleaning ?t - tool)
    (dirty ?r - room)
    (tidy ?it - item)
  )

  (:action move
    :parameters (?bot - robot ?from - room ?to - room)
    :precondition (and (at ?bot ?from) (connected ?from ?to))
    :effect (and (not (at ?bot ?from)) (at ?bot ?to))
  )

  (:action pickup-item
    :parameters (?bot - robot ?it - item ?r - room)
    :precondition (and (at ?bot ?r) (at ?it ?r) (handfree ?bot))
    :effect (and (holding ?bot ?it) (not (at ?it ?r)) (not (handfree ?bot)))
  )

  (:action pickup-from-surface
    :parameters (?bot - robot ?it - item ?s - surface ?r - room)
    :precondition (and (at ?bot ?r) (surface-in ?s ?r) (on ?it ?s) (handfree ?bot))
    :effect (and (holding ?bot ?it) (not (on ?it ?s)) (not (at ?it ?r)) (not (handfree ?bot)))
  )

  (:action place-on-surface
    :parameters (?bot - robot ?it - item ?s - surface ?r - room)
    :precondition (and (at ?bot ?r) (surface-in ?s ?r) (holding ?bot ?it))
    :effect (and (on ?it ?s) (at ?it ?r) (not (holding ?bot ?it)) (handfree ?bot) (tidy ?it))
  )

  (:action pickup-tool
    :parameters (?bot - robot ?t - tool ?r - room)
    :precondition (and (at ?bot ?r) (at ?t ?r) (handfree ?bot))
    :effect (and (holding ?bot ?t) (not (at ?t ?r)) (not (handfree ?bot)))
  )

  (:action putdown-tool
    :parameters (?bot - robot ?t - tool ?r - room)
    :precondition (and (at ?bot ?r) (holding ?bot ?t))
    :effect (and (at ?t ?r) (not (holding ?bot ?t)) (handfree ?bot))
  )

  (:action putdown-item
    :parameters (?bot - robot ?it - item ?r - room)
    :precondition (and (at ?bot ?r) (holding ?bot ?it))
    :effect (and (at ?it ?r) (not (holding ?bot ?it)) (handfree ?bot))
  )

  (:action clean-room
    :parameters (?bot - robot ?r - room ?t - tool)
    :precondition (and (at ?bot ?r) (holding ?bot ?t) (tool-for-cleaning ?t) (dirty ?r))
    :effect (and (not (dirty ?r)))
  )
)
