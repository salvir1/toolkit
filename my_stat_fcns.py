import numpy as np
from scipy import stats


def welch_test_statistic(sample_1, sample_2):
    '''
    Computes the t-statistic for two sample arrays with different but normally distributed variances.
    Parameters:
    sample_1: numpy array
    sample_2: numpy array

    Returns:
    t-statistic
    '''
    numerator = np.mean(sample_1) - np.mean(sample_2)
    denominator_sq = (np.var(sample_1) / len(sample_1)) + (np.var(sample_2) / len(sample_2))
    return numerator / np.sqrt(denominator_sq)

def welch_satterhwaithe_df(sample_1, sample_2):
    '''Calculate the degrees of freedom for a two-sample t-test.
    Parameters:
    sample_1: numpy array
    sample_2: numpy array

    Returns:
    degrees of freedom
    '''
    ss1 = len(sample_1)
    ss2 = len(sample_2)
    df = (
        ((np.var(sample_1)/ss1 + np.var(sample_2)/ss2)**(2.0)) / 
        ((np.var(sample_1)/ss1)**(2.0)/(ss1 - 1) + (np.var(sample_2)/ss2)**(2.0)/(ss2 - 1))
    )
    return df

def bootstrap_sample_means(data, n_bootstrap_samples=200):
    '''
    Generates an array of bootstrap sample means. Each bootstramp sample has
    the same length as the given dataset. The given dataset is resampled with
    replacement to generate the bootstraps.
    Parameters:
    ----------
    data: an array of samples
    n_bootstrap_samples: Number of bootstrap samples to generate

    Returns:
    -------
    An array of bootstrap sample means
    '''
    bootstrap_sample_means = []
    for i in range(n_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_sample_means.append(np.mean(bootstrap_sample))
    return bootstrap_sample_means

def compute_power(n, sigma, alpha, mu0, mua):
    '''
    Computes the power for an a/b test
    Parameters:
    ----------
    n: size of the sample
    sigma: population standard deviation
    mu0: population mean
    mua: effect
    '''
    
    standard_error = sigma / n**0.5
    h0 = stats.norm(mu0, standard_error)
    ha = stats.norm(mua, standard_error)
    critical_value = h0.ppf(1 - alpha)
    power = 1 - ha.cdf(critical_value)
    return power