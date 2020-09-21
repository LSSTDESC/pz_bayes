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
    # Doesn't work when data is a single row
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
            #dx = data[:, :, np.newaxis] - self._weights
            #distsq = np.sum(dx ** 2, axis=1)
            #bmu = np.argmin(distsq, axis=1)
            bmu = np.empty(len(data), dtype=int)
            for i in range(len(data)):
                dx = data[i].reshape(-1,1) - self._weights
                distsq = np.sum(dx ** 2, axis=0)
                bmu[i] = np.argmin(distsq)
        # Calculate best-matching cell for a single input:
        elif len(data.shape) == 1:
            dx = data.reshape(-1, 1) - self._weights
            distsq = np.sum(dx ** 2, axis=0)
            bmu = np.argmin(distsq)
        # Find the map site with the smallest distance (largest dot product).
        if return_distances: return(bmu, dx)
        else: return(bmu)

        
    def fit(self, data, target, maxiter=100, eta=0.5, init='random', seed=123, somz=False):
        
        rng = np.random.RandomState(seed)

        self.data = data
        self.target = target

        # Reformat data if not a numpy array.        
        if type(self.data) is np.ndarray:
            pass
        else:   
            self.data = table_to_array(self.data)
        
        N, D = self.data.shape

        # Store loss values for every epoch.
        self._loss = np.empty(maxiter)
        if init == 'random':
            sigmas = np.std(self.data, axis=0)
            if somz:
                self._weights = (rng.rand(D, self._mapgeom.size)) + data[0][0]
            else:
                self._weights = sigmas.reshape(-1, 1) * rng.normal(size=(D, self._mapgeom.size))
            
        else:
            raise ValueError('Invalid init "{}".'.format(init))

        if somz:
            tt = 0
            sigma0 = np.max(self._mapgeom.separations)
            sigma_single = np.min(self._mapgeom.separations[np.where(self._mapgeom.separations > 0.)])
            aps = 0.8
            ape = 0.5
            nt = maxiter * N
            for it in range(maxiter):
                loss = 0.
                alpha = aps * (ape / aps) ** (tt / nt)
                sigma = sigma0 * (sigma_single / sigma0) ** (tt / nt)
                index_random = rng.choice(N, N, replace=False)
                for i in range(N):
                    tt += 1
                    inputs = self.data[index_random[i]]
                    best = self.find_bmu(inputs)
                    h = np.exp(-(self._mapgeom.separations[best] ** 2) / sigma ** 2)
                    dx = inputs.reshape(-1, 1) - self._weights
                    loss += np.sqrt(np.sum(dx ** 2, axis=0))[best]
                    self._weights += alpha * h * dx
                self._loss[it] = loss
        else:
            # Randomize data
            rndm = rng.choice(np.arange(N), size=N, replace=False)
            data = self.data[rndm]
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
        
        ## TO DO: need to handle empty cells.
        ## Find cell each training vector belongs to
        self._indices = self.find_bmu(self.data)
        ## Get distribution of feature values for each cell
        self._feature_dist = [self.data[self._indices == i] for i in range(self._mapgeom.size)]
        self._target_dist = [target[self._indices == i] for i in range(self._mapgeom.size)]
        ## Should be mean or median?
        self._target_vals = [np.mean(self._target_dist[i]) for i in range(self._mapgeom.size)]
        self._target_pred = np.array(self._target_vals)[self._indices]


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

    def map_to_som(self, data):
    
        '''Takes input data of shape (N, features) and returns the predicted redshifts.'''
            
        # Reformat data if not a numpy array.        
        if type(data) is np.ndarray:
            pass
        else:   
            data = table_to_array(data)
            
        ## Calculate distance between data weights and SOM weights to find 
        ## best-matching cell for each input vector.
        best = self.find_bmu(data)
        ## Mean redshift per cell
        vals = np.array(self._target_vals)
        return(vals[best])

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

    def plot_statistic(self, feature=None, statistic=np.nanmean, return_stat=False):

        ## To do: handle empty cells

        if feature:
            fig, axs = plt.subplots(1,2, figsize=(12,5))
            axs = axs.ravel()
            # Plot statistic of feature per cell
            stat = np.asarray([statistic(self._feature_dist[i][:,feature]) for i in range(self._mapgeom.size)])
            im0 = axs[0].imshow(stat.reshape(self._mapgeom.shape), origin='lower', interpolation='none', cmap='viridis')
            fig.colorbar(im0, ax=axs[0])
            # Plot statistic of difference between feature weights and node weights per cell
            diff = np.asarray([statistic(self._feature_dist[i] - self._weights.T[i], axis=0)[feature] for i in range(self._mapgeom.size)])
            im1 = axs[1].imshow(diff.reshape(self._mapgeom.shape), origin='lower', interpolation='none', cmap='viridis')
            fig.colorbar(im1, ax=axs[1])
            plt.show()

        else:
            stat = np.asarray([statistic(self._target_dist[i]) for i in range(self._mapgeom.size)])
            plt.figure(figsize=(10,7))
            plt.imshow(stat.reshape(self._mapgeom.shape), origin='lower', interpolation='none', cmap='viridis')
            plt.colorbar()
            plt.show()
            
        if return_stat:
            return(stat)

    def build_density(self, data, target, nbins=50):
        
        bins = np.linspace(0, 3, nbins + 1)
        density = np.zeros((nbins, nbins))

        train_dist = self._target_dist
        test_dat = table_to_array(data)
        best = self.find_bmu(test_dat)
        test_dist = [target[best == i] for i in range(self._mapgeom.size)]

        for cell, dist in enumerate(train_dist):
            if dist.size == 0:
                pass
            else:
                test_hist, _ = np.histogram(test_dist[cell], bins)
                train_rho, _ = np.histogram(dist, bins, density=True)
                for zbin, nz in enumerate(test_hist):
                    density[:, zbin] += nz * train_rho
        return(density)
    
    def plot_sed(self, table, cell):
        
        colnames = []
        for col in table.colnames:
            if 'sed' in col:
                colnames.append(col)
                    
        in_cell = table[self._indices == cell]
        if len(in_cell) == 0:
            return('No galaxies were mapped to this cell.')
        rnd = rng.choice(len(in_cell), size=1)
        sed = in_cell[rnd]

        plt.figure(figsize(10,7))
        wlen = np.empty(len(colnames))
        mags = np.empty(len(colnames))
        for k, sed_col in enumerate(colnames):
            to_jy = 1 / (4.4659e13 / (8.4 ** 2))
            jy = sed[sed_col] * to_jy
            ab = -2.5 * np.log10(jy / 3631)
            start, width = colnames[k].split('_')[1:]
            start, width = int(start), int(width)
            wlen[k] =  (start + (start + width)) / 2 # angstroms
            mags[k] = ab
        x = cell % np.abs(self._mapgeom.shape[0])
        y = cell // np.abs(self._mapgeom.shape[1])
        t = (f'Cell # {cell}, x = {x}, y = {y} \n'
             f'Photo-z estimate: {np.round(self._target_pred[cell], 3)}\n'
             f'# Objects in cell: {len(in_cell)}')
        plt.plot(wlen, mags, 'ro')
        plt.axis([500, 18500, np.min(mags) - 0.5, np.max(mags) + 0.5])
        plt.text(10000, np.mean(mags) + 0.5, t, ha='left', wrap=True)
        plt.gca().invert_yaxis()
        plt.xlabel(r'$\AA$')
        plt.ylabel(r'$m_{AB}$')
        plt.show()
                    
        





