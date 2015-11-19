import lasagne
import theano.tensor as T
import theano
from ntm import NTM


class NTMTraining:
	def __init__(self, inpbatch, inplength):
		self.inp = lasagne.layers.InputLayer(shape=(inpbatch, inplength))

	def addNTM(self, ntmmodel):
		self.ntm_model = NTM(self.inp)

	def train(self, rate=.001, grad_clip=100, num_epoch=100, batch_size=100):
		target_values = T.vector('target_output')
		output = lasagne.layers.get_output(self.ntm_model)
		flat = output.flatten()
		cost = ((flat * target_values)**2)
		allparams = lasagne.layers.get_all_params(output)
		updates = layers.updates.adadelta(cost, allparams, rate)
		print("#Start training:")

		for epoch in range(num_epoch):
			train(X,y, m)
