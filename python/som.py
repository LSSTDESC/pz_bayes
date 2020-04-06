import abc
import numpy as np
import matplotlib.pyplot as plt

class MapGeometry(abc.ABC):

    def __init__(self, size):
        self._size = size
        self._separations = None

    @property
    def size(self):
        """The flattened size of this map.
        """
        return self._size

    @property
    def separations(self):
        """The matrix of pairwise separations between map nodes.

        Uses lazy evaluation.  The matrix might be sparse.
        """
        if self._separations is None:
            self._separations = self._calculate_separations()
        return self._separations

    @abc.abstractmethod
    def _calculate_separations(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def plot(self, values, ax=None, **kwargs):
        """Plot a representation of the specified values for this map.

        Parameters
        ----------
        values : array
            1D array of length :attr:`size`
        ax : matplotlib axis or None
            Plot axis to use, or create a default axis when None.
        kwargs : dict
            Additional plot keyword arguments to pass to the implementation.
        """
        raise NotImplementedError()


class Grid(MapGeometry):
    
    def __init__(self, *signature, metric='L2'):
        """Create a rectilinear grid map geometry.
        
        The grid shape is specified by the absolute values of the signature.        
        A negative value in the signature indicates that distances along the
        corresponding axis are calculated using wrap around.
        """
        
        shape = [abs(k) for k in signature]
        x = np.empty(shape=shape)
        super(Grid, self).__init__(x.size)
        self._shape = x.shape
        self._wrap = [k < 0 for k in signature]
        if metric not in ('L0', 'L1', 'L2'):
            raise ValueError('Invalid metric "{metric}", should be one of L0, L1, L2.')
        self._metric = metric
        
    @property
    def shape(self):
        return self._shape
    
    @property
    def wrap(self):
        return self._wrap
    
    @property
    def metric(self):
        return self._metric
    
    def _calculate_separations(self):
        ndim = len(self.shape)
        separation = np.zeros(self.shape + self.shape)
        # Loop over axes.
        for k, wrapk in enumerate(self._wrap):
            nk = self.shape[k]
            xk = np.arange(nk)
            # Calculate the the (nk, nk) matrix of absolute integer separations along the k-th axis.
            dxk = np.abs(xk.reshape(nk, 1) - xk)
            if wrapk:
                W = dxk > nk // 2
                dxk[W] *= -1
                dxk[W] += nk
            # Add this axis' contribution to the separation matrix.
            S = [(nk if (j % ndim == k) else 1) for j in range(2 * ndim)]
            dxk = dxk.reshape(S)
            if self._metric == 'L2':
                separation += dxk ** 2
            elif self._metric == 'L1':
                separation += dxk
            elif self._metric == 'L0':
                separation = np.maximum(separation, dxk)
        if self._metric == 'L2':
            # Take sqrt in place.
            np.sqrt(separation, out=separation)
        return separation.reshape(self.size, self.size)
    
    def plot(self, values, ax=None, **kwargs):
        """Plot an image of the input values.
        
        If the grid dimension is > 2, show a slice in the largest
        2 dimensions.
        """
        ndim = len(self.shape)
        if ndim == 1:
            # Plot a single row of values.
            values = values.reshape(1, self.size)
        elif ndim > 2:
            # Plot a slice in the largest 2 dimensions.
            largest = np.argsort(self.shape)[-2:]
            idx = [slice(None) if k in largest else 1 for k in range(ndim)]
            values = values[idx]
        ax = ax or plt.gca()
        ny, nx = values.shape
        ax.imshow(values, interpolation='none', origin='lower',
                  extent=[-0.5, nx - 0.5, -0.5, ny - 0.5], **kwargs)
        ax.axis('off')

class SelfOrganizingMap(object):
    
    def __init__(self, mapgeom):
        self._mapgeom = mapgeom
        
    def fit(self, data, maxiter=100, eta=0.5, sigma_frac=1, init='random', seed=123):

        # Reformat data if not a numpy array.        
        if type(data) is np.ndarray:
            pass
        else:
            colnames = data.colnames

            data_arr = np.zeros((len(data),len(colnames)))
            for k, name in enumerate(colnames):
                data_arr[:,k] = data[name]
                
            data = data_arr
        
        N, D = data.shape
        self._dx = np.empty((maxiter, N, D, self._mapgeom.size))
        self._corrections = np.empty((maxiter, N, D, self._mapgeom.size))
        # Store BMU for every epoch, and after the introduction of each new data vector.
        self._winner = np.empty((maxiter, N), np.uint32)
        # Store loss values for every epoch.
        self._loss = np.empty(maxiter)
        # Store the state of all weights for each epoch, and after the introduction of 
        # each new data vector.
        self._all_weights = np.empty((maxiter, N, D, self._mapgeom.size))
        rng = np.random.RandomState(seed)
        if init == 'random':
            sigmas = np.std(data, axis=0)
            # Real-time value of weights.
            self._weights = sigmas.reshape(-1, 1) * rng.normal(size=(D, self._mapgeom.size))
        else:
            raise ValueError('Invalid init "{}".'.format(init))
        # Calculate mean separation between grid points as a representative large scale.
        #large_scale = np.mean(self._mapgeom.separations)
        # Make sure the initial neighborhood is at least as large as the largest separation. 
        large_scale = np.max(self._mapgeom.separations)
        # Calculate the mean separation between N uniformly distributed points in D dimensions
        # as a representative small scale.
        volume = np.prod(self._mapgeom.shape)
        small_scale = (volume / N) ** (1 / D)
        assert small_scale < large_scale, 'Check the scales!'
        scale = sigma_frac * large_scale
        self._scale = scale
        #dscale = (small_scale / large_scale) ** (1 / maxiter)
        for i in range(maxiter):
            loss = 0.
            learn_rate = eta ** (i / maxiter)
            gauss_width = scale ** (1 - i / maxiter)
            for j, x in enumerate(data):
                # Calculate the Euclidean data-space distance squared between x and
                # each map site's weight vector.
                dx = x.reshape(-1, 1) - self._weights
                self._dx[i,j] = dx 
                distsq = np.sum(dx ** 2, axis=0)
                # Find the map site with the smallest distance (largest dot product).
                self._winner[i,j] = k = np.argmin(distsq)
                # The loss is the sum of smallest (data space) distances for each data point.
                loss += np.sqrt(distsq[k])
                # Update all weights (dz are map-space distances).
                dz = self._mapgeom.separations[k]
                self._corrections[i,j] = learn_rate * np.exp(-0.5 * (dz / gauss_width) ** 2) * dx
                self._weights += learn_rate * np.exp(-0.5 * (dz / gauss_width) ** 2) * dx
                self._all_weights[i,j] = self._weights
            self._loss[i] = loss


def u_matrix(som_weights):
    
    ''' * Add option to interpolate onto finer grid
    From p. 337 of this paper https://link.springer.com/content/pdf/10.1007%2F978-3-642-15381-5.pdf'''
    
    rows, cols, D = som_weights.shape
    
    u_matrix = np.empty((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            dist = 0
            ## neighbor above
            if i < rows - 1:
                dist += np.sqrt(np.sum((som_weights[i,j] - som_weights[i+1,j]) ** 2))
            ## neighbor below
            if i > 0:
                dist += np.sqrt(np.sum((som_weights[i,j] - som_weights[i-1,j]) ** 2))
            ## neighbor left
            if j > 0:
                dist += np.sqrt(np.sum((som_weights[i,j] - som_weights[i,j-1]) ** 2))
            ## neighbor right
            if j < cols - 1:
                dist += np.sqrt(np.sum((som_weights[i,j] - som_weights[i,j+1]) ** 2))
            u_matrix[i,j] = np.sum(dist)
            
    return(u_matrix)

def make_som(data, nmap=-50, niter=100, eta=0.5, sigma_frac=1, rgb=None, seed=123):

    # Build the self-organizing map.
    som = SelfOrganizingMap(Grid(nmap, nmap))
    som.fit(data, maxiter=niter, eta=eta, sigma_frac=sigma_frac, seed=seed)

    if type(data) is np.ndarray:
        N, D = data.shape
    else:
        D = len(data.colnames)

    # Plot the results.
    fig, ax = plt.subplots(2, 1, figsize=(8, 16))
    ax = ax.ravel()
    weights = som._weights.T
    # Normalize weights to be between [0,1]
    weights = (weights - weights.min(axis=0)) / (weights.max(axis=0) - weights.min(axis=0))
    if rgb:
        assert len(rgb) == 3
        # Select features to show in RGB map
        rgb_weights = weights[:,rgb]
        rgb_map = rgb_weights.reshape(abs(nmap), abs(nmap), 3)
        im = ax[0].imshow(rgb_map, interpolation='none', origin='lower', cmap='viridis')
    else:
        u_map = u_matrix(weights.reshape(abs(nmap), abs(nmap), D))
        im = ax[0].imshow(u_map, interpolation='none', origin='lower', cmap='viridis')
    ax[0].axis('off')
    ax[1].plot(som._loss, 'ko')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Training Loss')
    plt.tight_layout()

    # Return SOM and normalized weights
    return(som, weights)

def deep_phot_hist2d(deep_data, phot_data, cols=['u-g', 'g-r'], bins=100):
    
    n_deep= len(deep_data)
    n_phot = len(phot_data)
    
    fig, axs = plt.subplots(1,2, figsize=(16,6))
    axs = axs.ravel()

    hist2d0 = np.histogram2d(deep_data[cols[0]], deep_data[cols[1]], bins=bins)
    hist2d1 = np.histogram2d(phot_data[cols[0]], phot_data[cols[1]], bins=bins)

    im0 = axs[0].hist2d(deep_data[cols[0]], deep_data[cols[1]], weights=np.ones(n_deep) / np.max(hist2d0[0]), bins=bins)
    cbar0 = fig.colorbar(im0[-1], ax=axs[0])
    cbar0.set_label('# / cell max', rotation=270, labelpad=20)
    axs[0].set_title('rndm_spec histogram')

    im1 = axs[1].hist2d(phot_data[cols[0]], phot_data[cols[1]], weights=np.ones(n_phot) / np.max(hist2d1[0]), bins=bins)
    cbar1 = fig.colorbar(im1[-1], ax=axs[1])
    cbar1.set_label('# / cell max', rotation=270, labelpad=20)
    axs[1].set_title('rndm_phot histogram')

    plt.show()

def map_phot_to_som(data, normalized_weights):
    
    '''Takes an (N, features) data array and a SOM and returns
    the flattened SOM index to which each data vector belongs, 
    as well as the number of data points mapped to each SOM cell.'''
        
    if type(data) is np.ndarray:
            pass
    else:
        colnames = data.colnames

        data_arr = np.zeros((len(data),len(colnames)))
        for k, name in enumerate(colnames):
            data_arr[:,k] = data[name]

        data = data_arr

    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

    N, D = data.shape
    
    #assert normalized_weights.shape[0] == som._mapgeom.size
    #assert normalized_weights.shape[1] == D
        
    ## Calculate distance between data weights and SOM weights
    som_indices = np.empty(N, dtype=int)
    for i, x in enumerate(data):
        dx = (x - normalized_weights)
        som_indices[i] = np.argmin(np.sum(dx ** 2, axis=1))
        
    ## Determine frequency of each index on SOM resolution grid
    som_counts = np.bincount(som_indices, minlength=(normalized_weights.shape[0]))
    
    return(som_indices, som_counts)

def plot_counts_per_cell(data, normalized_weights, norm=None):
    
    '''Plot number of data points mapped to each SOM cell

    Counts must have same shape as SOM grid.'''

    indices, counts = map_phot_to_som(data, normalized_weights)
    nside = int(np.sqrt(normalized_weights.shape[0]))

    plt.figure(figsize=(10,7))
    plt.imshow(counts.reshape(nside,nside), origin='lower', interpolation='none', 
        cmap='viridis', norm=norm)
    plt.colorbar()
    plt.title('Number per SOM cell')
    plt.show()
    
    
def plot_statistic(som_indices, som_counts, target_feature, rows, cols, statistic=np.mean):
    
    stat = np.asarray([statistic(target_feature[som_indices == i]) for i in range(rows * cols)])
    
    plt.figure(figsize=(10,7))
    plt.imshow(stat.reshape(rows,cols), origin='lower', interpolation='none', cmap='viridis')
    plt.colorbar()
    plt.title('{} of target feature per cell'.format(statistic.__name__))
    plt.show()
    

def plot_dist_in_cell(som_indices, target_feature, cols, idx=(0,0), bins=20):
        
    flattened_idx = idx[0] * cols + idx[1]
    
    plt.figure(figsize=(10,7))
    plt.hist(target_feature[som_indices == flattened_idx], bins=bins)
    plt.title('n(z) in bin {}'.format(idx))
    plt.xlabel('z')
    plt.ylabel('counts')
    plt.show()
    
    
def plot_sed_in_cell(data, som_indices, cols, rng, idx=(0,0)):
    
    if type(data) is np.ndarray:
            pass
    else:
        colnames = data.colnames

        data_arr = np.zeros((len(data),len(colnames)))
        for k, name in enumerate(colnames):
            data_arr[:,k] = data[name]

        data = data_arr
    
    
    flattened_idx = idx[0] * cols + idx[1]
    gals_in_cell = data[som_indices == flattened_idx]
    rndm_idx = rng.randint(low=0, high=len(gals_in_cell), size=1)
    
    plt.figure(figsize=(10,7))
    plt.plot(gals_in_cell[rndm_idx].reshape(-1), '.')
    plt.title('SED in bin {}'.format(idx))
    plt.xlabel('band')
    plt.ylabel('mag')
    plt.show()





