import numpy as np
import os
import sys
import struct
from sklearn.metrics import classification_report
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--x_train_dir', help='Path to X_train.')
parser.add_argument('--y_train_dir', help='Path to y_train.')
parser.add_argument('--model_output_dir', help='Path to godlike model.')

args = parser.parse_args()

### Load data
def read_idx(filename):
	with open(filename, 'rb') as fp:
		zero, data_type, dims = struct.unpack('>HBB', fp.read(4))
		shape = tuple(struct.unpack('>I', fp.read(4))[0] for d in range(dims))
		return np.frombuffer(fp.read(), dtype=np.uint8).reshape(shape)

def load_dataset(dtype):
	assert dtype in ('train', 'test')

	if dtype == 'train':
		samples_file = args.x_train_dir
		labels_file = args.y_train_dir
	else:
		samples_file = args.x_test_dir
		labels_file = args.y_test_dir

	X = read_idx(samples_file)
	y = read_idx(labels_file)

	return (X, y)

### Prepare data
def to_float32(X):
	X = X.astype('float32')
	return X

def add_bias(X):
	bias_column = np.ones((X.shape[0], 1), dtype='float32')
	X = np.hstack((bias_column, X))
	return X

def scale_data(X, norms=None):
	if norms is None:
		norms = np.amax(X, 0)
	norms[norms == 0] = 1
	X /= norms[np.newaxis, :]

	return X, norms

def flatten_matrix(X):
	n_samples = X.shape[0]
	X = X.reshape((n_samples, -1))

	return X

def prepare_data(X):
	X = to_float32(X)
	X = flatten_matrix(X)
	X, _ = scale_data(X)

	return X

### Batching
def get_batch(array, batch_idx):
	start = batch_idx * batch_size
	stop = (batch_idx + 1) * batch_size

	return array[start: stop]

def load_batches(dtype):
	assert dtype in ('train', 'test')

	print '\nLoading batches...'

	if dtype == 'train':
		n_batches = 12
	else:
		n_batches = 2

	path = 'batches_' + dtype + '/batch_'

	batches = []
	for i in xrange(n_batches):
		batches.append(np.load(path + str(i)))

	kernel_norms = compute_kernel_norms(batches)
	make_kernel_scaling(batches, kernel_norms)

	return batches

### Kernel
def compute_simularity_matrix(batch, landmarks):
	n_features = batch.shape[1]

	simularity_matrix = (1. / n_features * np.matmul(batch, landmarks.T)) ** 5

	return simularity_matrix

def choose_landmarks():
	np.random.seed(220)
	landmarks_idx = np.random.choice(xrange(60000), size=10000, replace=False)
	return landmarks_idx

def kernelize(X, landmarks, dtype):
	# the only option for fast matmul on 64-bit kofevarka
	assert X.dtype == 'float32'
	assert dtype in ('train', 'test')

	if dtype == 'train':
		n_samples = 60000
	else:
		n_samples = 10000
	assert X.shape[0] == n_samples

	batch_num = n_samples // batch_size

	directory = 'batches_' + dtype
	if not os.path.exists(directory):
		os.makedirs(directory)

	print '\n' \
		  'Kernelizing ' + str(batch_num) + ' batches...\n' \
		  'Done: ',

	for idx in xrange(batch_num):
		batch = get_batch(X, idx)
		sm = compute_simularity_matrix(batch, landmarks)

		sm = add_bias(sm)

		with open(directory + '/batch_' + str(idx), 'wb') as fp:
			np.save(fp, sm)

		print str(idx),
	print ''

	return

def compute_kernel_norms(batches):
	# get norms over all batches
	norms = np.amax(batches[0], 0)

	for i in xrange(1, len(batches)):
		new_norms = np.amax(batches[i], 0)
		norms_arr = np.vstack((norms, new_norms))
		norms = np.amax(norms_arr, 0)

	return norms

def make_kernel_scaling(batches, norms):
	# scale all batches
	for i in xrange(len(batches)):
		batches[i], _ = scale_data(batches[i], norms)

	return

### Model
def predict_score(X, W):
	return np.matmul(X, W)

def predict_labels(batches, W):
	predicted_scores = np.empty((0, 10))

	for idx, batch in enumerate(batches):
		batch_score = predict_score(batch, W)
		predicted_scores = np.concatenate((predicted_scores, batch_score))

	predicted_labels = np.array(map(lambda x: np.argmax(x), predicted_scores))

	return predicted_labels

def init_weights(n_features, n_classes, a, b):
	return a + (b - a) * np.random.rand(n_features, n_classes)

### Estimate Record Error
def svm_loss(W, X, y, reg):
	n_samples = X.shape[0]

	loss = 0.0

	scores = np.matmul(X, W)
	correct_class_score = scores[np.arange(n_samples), y]
	margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1)
	margins[np.arange(n_samples), y] = 0
	loss = np.sum(margins)
	loss += 0.5 * reg * np.sum(np.square(W[1:]))

	loss /= n_samples

	return loss

def svm_grad(W, X, y, reg):
	n_samples = X.shape[0]

	dW = np.zeros(W.shape)

	# compute margins
	scores = np.matmul(X, W)
	correct_class_score = scores[np.arange(n_samples), y]
	margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1)
	margins[np.arange(n_samples), y] = 0

	# compute grad
	mask = np.zeros(margins.shape)
	mask[margins > 0] = 1
	np_sup_zero = np.sum(mask, axis=1)
	mask[np.arange(n_samples), y] = -np_sup_zero
	dW = np.matmul(X.T, mask)

	dW /= n_samples

	unbiased_W = W.copy()
	unbiased_W[0] = 0
	dW += reg * unbiased_W

	return dW

def compute_total_loss(loss_fn, W, batches, y, lambda_coef):
	total_loss = 0
	for idx, batch_X in enumerate(batches):
		batch_y = get_batch(y, idx)
		batch_loss = loss_fn(W, batch_X, batch_y, lambda_coef)
		total_loss += batch_loss
	return total_loss

def get_batch_ids(sample_id):
	batch_id = sample_id // batch_size
	sample_local_id = sample_id % batch_size
	return (batch_id, sample_local_id)

def sampling_mbatch(batches, y, n_samples):
	mbatch_ids = np.random.choice(xrange(n_samples), size=mbatch_size, \
								  replace=False)

	mbatch_X = np.array(
		[batches[batch_id][sample_local_id] for (batch_id, sample_local_id) in map(get_batch_ids, mbatch_ids)])
	mbatch_y = y[mbatch_ids]

	return (mbatch_X, mbatch_y)

def sgd(loss_fn, grad_fn, batches, y, W, mbatch_size, learning_rate, iter_num, \
		lambda_coef, verbose=False):
	_, n_features = batches[0].shape
	n_samples = len(batches) * batch_size

	forget_rate = float(mbatch_size) / n_samples

	aiming = False

	# init loss
	last_loss = float('+inf')
	current_loss = compute_total_loss(loss_fn, W, batches, y, lambda_coef)

	# sampling batch
	mbatch_X, mbatch_y = sampling_mbatch(batches, y, n_samples)

	# compute grad
	grad = grad_fn(W, mbatch_X, mbatch_y, lambda_coef)

	for i in range(iter_num):

		# make step
		W -= learning_rate * grad

		# est Q
		mbatch_loss = loss_fn(W, mbatch_X, mbatch_y, lambda_coef)
		# stash Q
		current_loss = (1 - forget_rate) * current_loss + forget_rate * mbatch_loss

		if current_loss < last_loss:
			last_loss = current_loss
			learning_rate *= 1.05

			aiming = False

		else:
			# reverse step
			W += learning_rate * grad

			# reverse Q
			current_loss = (current_loss - forget_rate * mbatch_loss) / (1. - forget_rate)

			learning_rate /= 2
			aiming = True

		mbatch_X, mbatch_y = sampling_mbatch(batches, y, n_samples)

		grad = grad_fn(W, mbatch_X, mbatch_y, lambda_coef)

		if verbose == True:
			print 'iter', i, 'loss:', current_loss, 'alpha:', learning_rate, '!!!Aiming!!!' if aiming else ''

	return W

def train_classifier(batches, y, mbatch_size, learning_rate, iter_num, lambda_coef):
	_, n_features = batches[0].shape
	n_classes = 10

	W = init_weights(n_features, n_classes, -0.0001, 0.0001)

	W = sgd(svm_loss, svm_grad, batches, y, W, mbatch_size, learning_rate, iter_num, \
			lambda_coef, verbose=False)

	return W

### Code

# --- CONSTANTS ---
batch_size = 5000
# --- --------- ---

# check argv
if (args.x_train_dir is None or \
	args.y_train_dir is None or \
	args.model_output_dir is None):
	print '\n' \
		  'Specify train data and directory for model!\n'
	sys.exit()

# load train data
X, y = load_dataset('train')
X = prepare_data(X)

# kernelize train data
landmarks_idx = choose_landmarks()
kernelize(X, X[landmarks_idx], 'train')

# load kernel train data
batches_train = load_batches('train')

# Train do dima
mbatch_size = 100
learning_rate = 0.1
iter_num = 5000
lambda_coef = 0

print '\nTraining...'
W = train_classifier(batches_train, y, mbatch_size, learning_rate, iter_num, lambda_coef)

# Save weights
with open(args.model_output_dir, 'wb') as fp:
	np.save(fp, W)

print '\nDone. Path to trained model:\n' \
	  + args.model_output_dir

predicted_labels = predict_labels(batches_train, W)
print '\n', classification_report(y, predicted_labels)