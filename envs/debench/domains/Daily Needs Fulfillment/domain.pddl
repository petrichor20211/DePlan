;; Bucket: Daily Needs Fulfillment
(define (domain elderly-homecare-daily-needs)
  (:requirements :strips :typing)
  (:types
    robot person
    room
    item container appliance
  )
  (:predicates
    (at ?x - (either robot person item container appliance) ?r - room)
    (connected ?from - room ?to - room)
    (holding ?bot - robot ?it - (either item container))
    (handfree ?bot - robot)
    (is-cup ?c - container)
    (is-food ?f - item)
    (heated ?f - item)
    (contains-water ?c - container)
    (water-available ?r - room)
    (microwave ?m - appliance)
    (warmed-meal-delivered ?p - person)
    (water-delivered ?p - person)
  )

  (:action move
    :parameters (?bot - robot ?from - room ?to - room)
    :precondition (and (at ?bot ?from) (connected ?from ?to))
    :effect (and (not (at ?bot ?from)) (at ?bot ?to))
  )

  (:action pickup
    :parameters (?bot - robot ?x - (either item container) ?r - room)
    :precondition (and (at ?bot ?r) (at ?x ?r) (handfree ?bot))
    :effect (and (holding ?bot ?x) (not (at ?x ?r)) (not (handfree ?bot)))
  )

  (:action fill-cup-with-water
    :parameters (?bot - robot ?c - container ?r - room)
    :precondition (and (at ?bot ?r) (holding ?bot ?c) (is-cup ?c) (water-available ?r))
    :effect (contains-water ?c)
  )

  (:action deliver-water
    :parameters (?bot - robot ?c - container ?p - person ?r - room)
    :precondition (and (at ?bot ?r) (at ?p ?r) (holding ?bot ?c) (is-cup ?c) (contains-water ?c))
    :effect (and (water-delivered ?p) (at ?c ?r) (not (holding ?bot ?c)) (handfree ?bot))
  )

  (:action heat-food-in-microwave
    :parameters (?bot - robot ?f - item ?m - appliance ?r - room)
    :precondition (and (at ?bot ?r) (holding ?bot ?f) (is-food ?f) (at ?m ?r) (microwave ?m))
    :effect (heated ?f)
  )

  (:action deliver-heated-meal
    :parameters (?bot - robot ?f - item ?p - person ?r - room)
    :precondition (and (at ?bot ?r) (at ?p ?r) (holding ?bot ?f) (is-food ?f) (heated ?f))
    :effect (and (warmed-meal-delivered ?p) (at ?f ?r) (not (holding ?bot ?f)) (handfree ?bot))
  )
)
