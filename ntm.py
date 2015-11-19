import lasagne

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
		''' Provide controller output
		    And provides external output
		'''
		T.tanh(T.dot(h, Wk) + bk)
		T.dot(h, Wc) + bc

	def _location_w(self, wc, g, wprev, S):
		wg = g * wc + (1 - g) * wprev
		wt = (wg * S).sum(axis=1)



