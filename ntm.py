import lasagne
import theano.tensor as T
import theano

# Set NTM layer

class NTM(lasagne.layers.RecurrentLayer):
	def __init__(self, inp, inpdim, outdim, nslots):
		self.inp = inp
		self.inpdim = inpdim
		self.outdim = outdim
		self.nslots = nslots

	def _read(self, w, M):
		return T.sum(self.W*M)

	def _write(self, w, Mprev, M, e, a, mask):
		'''
		   e - earse vector
		   Mprev - memory vectors
		   a - add vector
		   w - weights
		'''
		mem1 = M* ((1 - w) * e)
		mem2 = mem1 * w * a
		return mask * mem2 + (1 - mask) * M

	def _step(self, X, M, Mprev, WR, WW):
		pass

	def _controller(self, hidden, Ws, bs, Wc, bc, Wk, bk):
		''' 
		    Controller network
		    Provide controller output
		    And provides external output
		    It can be used as several types of networks
		    Returns number of read and writes heads
		'''
		itemhead = T.nnet.softmax(T.tanh(T.dot(h, Ws) + bs))
		itemwrite = T.nnet.tanh(T.dot(h, Wc) + bc)
		itemshifts = T.nnet.relu(T.dot(h, Wk) + bk)
		return itemhead, itemwrite, itemshifts

	def _cosine(self, u, v):
		return u*v/(T.norm(u) * T.norm(v))

	def _content(self, u, v, beta):
		''' Content system
		'''
		return T.nnet.softmax(beta * self._cosine(u, v))

	def _location_w(self, wc, g, wprev, S):
		'''
		   g - interpolation gate \in (0,1)
		   wprev - weights on previous time-step
		   wc - weights produced by the content system

		   Output:
		     gated weights
		'''
		wg = g * wc + (1 - g) * wprev
		wt = (wg * S).sum(axis=1)

	def get_output_for(self, input):
		''' Main methiod for return result from NTM layer
		'''
		pass



