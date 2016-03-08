from algorithm.parameters import ELITE_SIZE,GENERATION_SIZE
from copy import copy

def generational_replacement(new_pop, individuals):
    """Return new pop. The ELITE_SIZE best individuals are appended
    to new pop if they are better than the worst individuals in new
    pop"""

    individuals.sort(reverse=True)
    for ind in individuals[:ELITE_SIZE]:
        new_pop.append(copy(ind))
    new_pop.sort(reverse=True)
    return new_pop[:GENERATION_SIZE]

#Provided but no flag set. Need to append code to use this
def steady_state_replacement(new_pop, individuals):
    """Return individuals. If the best of new pop is better than the
    worst of individuals it is inserted into individuals"""
    individuals.sort(reverse=True)
    individuals[-1] = max(new_pop + individuals[-1:])
    return individuals