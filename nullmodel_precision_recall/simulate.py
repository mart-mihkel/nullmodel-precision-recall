import numpy as np

from sklearn.metrics import precision_recall_curve
from plotnine import ggplot, aes, labs, geom_step


def __randomize_score(s, y=None, acc=0.0, tresh=0.5):
    """
    Randomly permute scores. Rig the scores for the model to have accuracy of
    at least `acc` with treshold of `tresh` if provided.
    """
    s = np.random.permutation(s)

    if y is None or acc == 0:
        return s

    n_rig = int(y.shape[0] * acc)
    n_neg = n_rig // 2
    n_pos = n_rig - n_neg

    yneg_i = np.random.choice(np.flatnonzero(y == 0), n_neg)
    ypos_i = np.random.choice(np.flatnonzero(y == 1), n_pos)

    sneg_i = np.flatnonzero(s < tresh)
    spos_i = np.flatnonzero(s >= tresh)

    neg = s[sneg_i]
    pos = s[spos_i]

    if neg.shape[0] < n_neg:
        neg = np.append(neg, np.repeat(1 - tresh, n_neg - neg.shape[0]))
    else:
        neg = np.random.choice(neg, n_neg)

    if pos.shape[0] < n_pos:
        pos = np.append(pos, np.repeat(tresh, n_pos - pos.shape[0]))
    else:
        pos = np.random.choice(pos, n_pos)

    s[yneg_i], s[ypos_i] = neg, pos

    return s


def __decreasing(p, r):
    """
    Pick points on precision recall curve where recall is decreasin.
    """
    idx = np.flip(np.diff(r[::-1], prepend=1.1) > 0)
    return p[idx], r[idx]


def simulate_nullmodels(n_sim: int,
                        n_samp: int,
                        ppos=0.5,
                        acc=0.0,
                        tresh=0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate nullmodels

    Parameters
    ----------
    n_sim : integer
        Number of simulations.
    n_samp : integer
        Number of samples in each simulation.
    ppos : float
        Percentage of positive labels in a sample.
        Must be between 0 and 1.
        Default is 0.5.
    acc : float
        Percentage of samples the nullmodel is rigged to label correctly with
        a treshold of `tresh`.
        Must be between 0 and 1.
        Defualt is 0.0.
    tresh : float
        Classification treshold when using `acc` to rig the model. Consider
        a data point positive when `y_score >= tresh`.
        Must be between 0 and 1.
        Defualt is 0.5.

    Returns
    -------
    (y_true, y_scores) : tuple
        Tuple of true labels with shape (n_samp,) and simulated nullmodel
        scores with shape (n_sim, n_samp).
    """
    n_pos = int(ppos * n_samp)

    y_true = np.repeat((0, 1), (n_samp - n_pos, n_pos))
    init_scores = np.tile(np.linspace(0, 1, num=n_samp), n_sim) \
                    .reshape((n_sim, n_samp))

    y_scores = [__randomize_score(s, y_true, acc, tresh) for s in init_scores]

    return y_true, np.array(y_scores)


def pr_curve_quantile(curves, q, n_knots=50) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute q-th quantile of precision recall curves.

    Parameters
    ----------
    curves : array
        Array of (precision, recall) tuples.
    q : float
        Quantile to compute.
    n_knots : integer
        Number of interpolation knots.
        Defualt is 50.

    Returns
    -------
    (precision, recall) : tuple
        Precision recall curve, with precision = q-th quantile of precisions
        and recall = interpolation knots.
    """
    knots = np.linspace(0, 1, num=n_knots)
    dec = [__decreasing(p, r) for p, r in curves]
    interps = [np.interp(knots, xp=r[::-1], fp=p)[::-1] for p, r in dec]
    return np.quantile(interps, q=q, axis=0), knots


def plot_simulations(n_sim: int,
                     n_samp: int,
                     ppos=0.5,
                     acc=0.0,
                     q=0.9,
                     tresh=0.5,
                     plot_all=False):
    """
    Simulate nullmodels and plot q-th quantile of results.

    Parameters
    ----------
    n_sim : integer
        Number of simulations.
    n_samp : integer
        Number of samples in each simulation.
    ppos : float
        Percentage of positive labels in a sample.
        Must be between 0 and 1.
        Default is 0.5.
    acc : float
        Percentage of samples the nullmodel is rigged to label correctly with
        a treshold of 0.5.
        Must be between 0 and 1.
        Defualt is 0.0.
    tresh : float
        Classification treshold when using `acc` to rig the model.
        Must be between 0 and 1.
        Defualt is 0.5.
    plot_all : boolean
        Wether to plot all simulations.
        Defualt is False.

    Examples
    -------
    >>> from nullmodel_precision_recall import plot_simulations
    >>> plot_simulations(100, 1000, ppos=0.5, acc=0.0)
    """
    y_true, y_scores = simulate_nullmodels(n_sim,
                                           n_samp,
                                           ppos=ppos,
                                           acc=acc,
                                           tresh=tresh)

    simulated_curves = [precision_recall_curve(y_true, y_s)[:2]
                        for y_s
                        in y_scores]

    pq, rq = pr_curve_quantile(simulated_curves, q)

    title = f'n_sim={n_sim} n_samp={n_samp} ppos={ppos} acc={acc} q={q}'
    g = ggplot() + labs(x='recall', y='precision', title=title)
    g = g + geom_step(aes(rq, pq))

    if not plot_all:
        g.show()
        return

    lens = [len(ps) for ps, _ in simulated_curves]
    group = np.repeat(np.arange(n_sim), lens)
    ps, rs = np.hstack(simulated_curves)

    g = g + geom_step(aes(rs, ps, group=group), alpha=1/n_sim)
    g.show()

