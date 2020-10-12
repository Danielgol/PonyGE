from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff



class conv(base_ff):

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):
    	print("HOUSTON "+ind.phenotype)
    	return 1

    	# olhar exemplo sequence_match
    	# pelo que aparenta, o ponyge gera uma sequencia de "palavras" pela gramática,
    	# depois nós pegamos essa sequencia e aplicamos como quisermos.
    	"""
    	total = 0
        matches = 0
        with torch.no_grad():
	        for batch in train_data:
	        	X, y = batch
	        	output = net(X.view(-1,28*28))
	        	for idx, i in enumerate(output):
			      if torch.argmax(i) == y[idx]:
			        matches += 1
			      total += 1

		accuracy = matches/total
		"""
		