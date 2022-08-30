from fitness.base_ff_classes.base_ff import base_ff
from difflib import SequenceMatcher

class string_match_daniel(base_ff):

    maximise = True
    multi_objective = True

    def __init__(self):
        super().__init__()
        #self.target = params['TARGET']
        self.num_obj = 1
        fit = base_ff()
        fit.maximise = True
        self.fitness_functions = [fit]
        self.default_fitness = [float('nan')]

    def evaluate(self, ind, **kwargs):

        guess = ind.phenotype

        print("Phenotype: "+guess)

        target = "Hello World"

        fitness = SequenceMatcher(None, guess, target).ratio()

        print("Fitness: "+str(fitness))

        return fitness

    
    @staticmethod
    def value(fitness_vector, objective_index):
        """
        This is a static method required by NSGA-II for sorting populations
        based on a given fitness function, or for returning a given index of a
        population based on a given fitness function.

        :param fitness_vector: A vector/list of fitnesses.
        :param objective_index: The index of the desired fitness.
        :return: The fitness at the objective index of the fitness vecror.
        """

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]
