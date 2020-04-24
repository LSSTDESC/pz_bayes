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

def table_to_array(data):

    colnames = data.colnames
    data_arr = np.zeros((len(data),len(colnames)))
    for k, name in enumerate(colnames):
        data_arr[:,k] = data[name]
    return(data_arr)

class SelfOrganizingMap(object):
    
    def __init__(self, mapgeom):
        self._mapgeom = mapgeom

    def find_bmu(self, data, return_distances=False):
        # Calculate best-matching cell for all inputs simultaneously:
        if len(data.shape) > 1:
            dx = data[:, :, np.newaxis] - self._weights
            distsq = np.sum(dx ** 2, axis=1)
            bmu = np.argmin(distsq, axis=1)
        # Calculate best-matching cell for a single input:
        elif len(data.shape) == 1:
            dx = data.reshape(-1, 1) - self._weights
            distsq = np.sum(dx ** 2, axis=0)
            bmu = np.argmin(distsq)
        # Find the map site with the smallest distance (largest dot product).
        if return_distances: return(bmu, dx)
        else: return(bmu)

        
    def fit(self, data, maxiter=100, eta=0.5, init='random', seed=123):
        
        # Reformat data if not a numpy array.        
        if type(data) is np.ndarray:
            pass
        else:   
            data = table_to_array(data)
        
        N, D = data.shape

        # Randomize data
        rng = np.random.RandomState(seed)
        rndm = rng.choice(np.arange(N), size=N, replace=False)
        data = data[rndm]

        # Store loss values for every epoch.
        self._loss = np.empty(maxiter)
        if init == 'random':
            sigmas = np.std(data, axis=0)
            self._weights = sigmas.reshape(-1, 1) * rng.normal(size=(D, self._mapgeom.size))
        else:
            raise ValueError('Invalid init "{}".'.format(init))
        # Calculate mean separation between grid points as a representative large scale.
        large_scale = np.mean(self._mapgeom.separations)
        for i in range(maxiter):
            loss = 0.
            learn_rate = eta ** (i / maxiter)
            gauss_width = large_scale ** (1 - i / maxiter)
            for j, x in enumerate(data):
                # Calculate the Euclidean data-space distance squared between x and
                # each map site's weight vector.
                bmu, dx = self.find_bmu(x, return_distances=True)
                distsq = np.sum(dx ** 2, axis=0)
                # The loss is the sum of smallest (data space) distances for each data point.
                loss += np.sqrt(distsq[bmu])
                # Update all weights (dz are map-space distances).
                dz = self._mapgeom.separations[bmu]
                self._weights += learn_rate * np.exp(-0.5 * (dz / gauss_width) ** 2) * dx
            self._loss[i] = loss


    def plot_u_matrix(self):
        
        ''' 
        Visualize the weights in two dimensions.
        
        * Add option to interpolate onto finer grid
        From p. 337 of this paper https://link.springer.com/content/pdf/10.1007%2F978-3-642-15381-5.pdf'''
        
        rows, cols = self._mapgeom.shape
        u_matrix = np.empty((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                dist = 0
                ## neighbor above
                if i < rows - 1:
                    dist += np.sqrt(np.sum((self._weights[i,j] - self._weights[i+1,j]) ** 2))
                ## neighbor below
                if i > 0:
                    dist += np.sqrt(np.sum((self._weights[i,j] - self._weights[i-1,j]) ** 2))
                ## neighbor left
                if j > 0:
                    dist += np.sqrt(np.sum((self._weights[i,j] - self._weights[i,j-1]) ** 2))
                ## neighbor right
                if j < cols - 1:
                    dist += np.sqrt(np.sum((self._weights[i,j] - self._weights[i,j+1]) ** 2))
                u_matrix[i,j] = np.sum(dist)

        plt.figure(figsize=(10,7))
        plt.imshow(u_map, interpolation='none', origin='lower', cmap='viridis')
        plt.show()

    def plot_rgb(self, features=None):

        '''Visualize the weights on an RGB scale using only three features.
        If features isn't specified, then the first three features are used.
        
        Inputs
        ------
        features: List of indices for each feature to include in the map.'''

        rows, cols = self._mapgeom.shape
        weights = self._weights.T
        # Normalize weights to be between [0,1]
        weights = (weights - weights.min(axis=0)) / (weights.max(axis=0) - weights.min(axis=0))
        # Select features to show in RGB map
        if features:
            rgb = weights[:,features]
        else:
            rgb = weights[:,:3]
        rgb_map = rgb.reshape(rows, cols, 3)

        plt.imshow(rgb_map, interpolation='none', origin='lower', cmap='viridis')
        plt.show()

    def map_to_som(self, data, target):
    
        '''Takes input data of shape (N, features) a trained SOM and returns
        the SOM index to which each input vector belongs and number of counts 
        per SOM cell.'''
            
        # Reformat data if not a numpy array.        
        if type(data) is np.ndarray:
            pass
        else:   
            data = table_to_array(data)

        if type(target) is np.ndarray:
            pass
        else:
            target = table_to_array(target)

        N, D = data.shape
        assert target.shape == (N,)
            
        ## Calculate distance between data weights and SOM weights
        self._indices, self._node_weight_separations = self.find_bmu(data, return_distances=True)
        ## Get distribution of feature values for each cell
        self._feature_dist = [data[self._indices == i] for i in range(self._mapgeom.size)]
        self._target_dist = [target[self._indices == i] for i in range(self._mapgeom.size)]

    def plot_counts_per_cell(self, norm=None):
    
        '''Plot number of data points mapped to each SOM cell.'''

        # Determine frequency of each index on SOM resolution grid
        counts = np.bincount(self._indices, minlength=(self._mapgeom.size))
        self._counts = counts.reshape(self._mapgeom.shape)

        plt.figure(figsize=(10,7))
        plt.imshow(self._counts, origin='lower', interpolation='none', 
            cmap='viridis', norm=norm)
        plt.colorbar()
        plt.title('Number per SOM cell')
        plt.show()

    def plot_statistic(self, feature=None, statistic=np.nanmean):

        ## To do: handle empty cells

        if feature:
            fig, axs = plt.subplots(1,2, figsize=(12,5))
            axs = axs.ravel()
            # Plot statistic of feature per cell
            stat = np.asarray([statistic(self._feature_dist[i][:,feature]) for i in range(self._mapgeom.size)])
            im0 = axs[0].imshow(stat.reshape(self._mapgeom.shape), origin='lower', interpolation='none', cmap='viridis')
            fig.colorbar(im0, ax=axs[0])
            # Plot statistic of difference between feature weights and node weights per cell
            diff = statistic(self._node_weight_separations, axis=0)[feature]
            im1 = axs[1].imshow(diff.reshape(self._mapgeom.shape), origin='lower', interpolation='none', cmap='viridis')
            fig.colorbar(im1, ax=axs[1])
            plt.show()

        else:
            stat = np.asarray([statistic(self._target_dist[i]) for i in range(self._mapgeom.size)])
            plt.figure(figsize=(10,7))
            plt.imshow(stat.reshape(self._mapgeom.shape), origin='lower', interpolation='none', cmap='viridis')
            plt.colorbar()
            plt.show()

    def plot_sed_in_cell(self, cell, seed=123):

        ngals = len(self._feature_dist[cell])

        if ngals == 0:
            return('No galaxies were mapped to this cell.')
        else:
            # Choose a random SED to plot
            rng = np.random.RandomState(seed=seed)
            idx = rng.randint(low=0, high=ngals, size=1)[0]
            plt.figure(figsize=(10,7))
            plt.plot(self._feature_dist[cell][idx], '.')
            plt.plot()
            plt.title('Random SED in bin {}'.format(cell))
            plt.xlabel('Filter')
            plt.ylabel('Magnitude')
            plt.show()





