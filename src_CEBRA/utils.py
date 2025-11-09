import numpy as np
import matplotlib.pyplot as plt

def match_sample_size_equal(a, b):
    """
    Downsamples the larger array to match the smaller one,
    using equally spaced selection along the first dimension.
    
    Parameters
    ----------
    a, b : np.ndarray
        Arrays to match along axis 0.
    
    Returns
    -------
    a_new, b_new : np.ndarray
        Arrays with the same number of rows (n_samples).
    """
    n_a, n_b = len(a), len(b)
    n_min = min(n_a, n_b)
    
    def downsample(x, target_len):
        n = len(x)
        if n == target_len:
            return x
        indices = np.linspace(0, n - 1, target_len, dtype=int)
        return x[indices]
    
    if n_a > n_min:
        a = downsample(a, n_min)
    if n_b > n_min:
        b = downsample(b, n_min)
    
    return a, b

def linear_fit(emb1, emb2):
    from sklearn.linear_model import LinearRegression
    fit = LinearRegression().fit(emb1, emb2)
    # coef, intercept = fit.coef_, fit.intercept_
    r_2 = fit.score(emb1, emb2)
    return r_2

def CKA(emb1, emb2):
    # centered kernel alignment based estimation of the correlation between the two embeddings
    from sklearn.metrics.pairwise import linear_kernel
    KX = linear_kernel(emb1, emb1)
    KY = linear_kernel(emb2, emb2)
    hsic = np.einsum('ij,ij', KX, KY) # same as np.sum(KX * KY) but much faster
    KX_norm = np.einsum('ij,ij', KX, KX)
    KY_norm = np.einsum('ij,ij', KY, KY)
    return hsic / np.sqrt(KX_norm * KY_norm)

class MetaParams(dict):

    def __init__(self, base, params_dict):
        super().__init__(params_dict)
        self.base = base

    def full_label(self):
        # generate label like "base_one0_two1_three2" where one two three are param short names, 0 1 and 2 are values
        label = self.base
        for key in self.keys():
            value = self.val(key)
            incl_name = self.include_param_name(key)
            if not incl_name:
                key = ''
            if isinstance(value, bool):
                if value:
                    label += f"_{key}"
            else:
                label += f"_{key}{value}"
        return label

    def val(self, param, default=None):
        if param not in self:
            return default
        return self.get(param)[0]

    def description(self, param, default=None):
        if param not in self:
            return None
        return self.get(param)[1]
    
    def include_param_name(self, param, default=None):
        if param not in self:
            return None
        all = self.get(param)
        if len(all) < 3:
            return True
        else:
            return all[2]

