import random
import mlx.core as mx

# Helpers to strictly adapt arrays to Python float lists for ribs
def _as_float_list(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [float(v) for v in x]

def _as_2d_float_list(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [[float(v) for v in row] for row in x]

# pip install ribs
import ribs

from . import autotune
from .autotune_global import AutotuneGlobal


class CMAMEOpt(autotune.Optimizer):
    """Quality Diversity optimize using CMA-ME to find a set of good and diverse hyperparameters"""

    def __init__(self, population=10, sigma=1.0, bins=15):
        """

        :param population: number of parameters to sample at once (scales linearly)
        :param sigma: initial variance along all dimensions
        :param bins: int or a Sequence[int] for each hyperparameter for the number of bins in the archive.
        More bins means more granularity along that dimension.
        """
        self.population = population
        self.sigma = sigma
        self.archive = None
        self.qd_score_offset = -3000
        self.num_emitters = 1
        self.bins = bins
        super().__init__()

    def setup_optimization(self):
        if not isinstance(self.tuner, AutotuneGlobal):
            raise RuntimeError(f"Quality diversity optimizers require global search space information provided "
                               f"by AutotuneMPPIGlobal")

        x = _as_float_list(self.tuner.flatten_params())
        ranges_dict = self.tuner.linearized_search_space()
        ranges = [(float(lo), float(hi)) for (lo, hi) in ranges_dict.values()]

        param_dim = len(x)
        bins = self.bins
        if isinstance(bins, (int, float)):
            bins = [int(bins) for _ in range(param_dim)]
        else:
            bins = [int(b) for b in bins]
        self.archive = ribs.archives.GridArchive(solution_dim=param_dim,
                                                 dims=bins,
                                                 ranges=ranges,
                                                 seed=random.randint(0, 10000), qd_score_offset=self.qd_score_offset)
        emitters = [
            ribs.emitters.EvolutionStrategyEmitter(self.archive, x0=_as_float_list(x), sigma0=self.sigma, batch_size=self.population,
                                                   seed=random.randint(0, 10000)) for i in
            range(self.num_emitters)
        ]
        self.optim = ribs.schedulers.Scheduler(self.archive, emitters)

    def optimize_step(self):
        if not isinstance(self.tuner, AutotuneGlobal):
            raise RuntimeError(f"Quality diversity optimizers require global search space information provided "
                               f"by AutotuneMPPIGlobal")

        params = _as_2d_float_list(self.optim.ask())
        # measure is the whole hyperparameter set - we want to diverse along each dimension

        cost_per_param = []
        all_rollouts = []
        bcs = []
        for param in params:
            full_param = self.tuner.unflatten_params(_as_float_list(param))
            res = self.tuner.evaluate_fn()
            # Mean cost using MLX
            cost_per_param.append(float(mx.mean(mx.array(res.costs))))
            all_rollouts.append(res.rollouts)
            behavior = self.tuner.linearize_params(full_param)
            bcs.append(_as_float_list(behavior))

        # Provide objectives as negatives of costs without NumPy
        objectives = [-float(c) for c in cost_per_param]
        self.optim.tell(objectives, _as_2d_float_list(bcs))

        best_param = self.archive.best_elite
        # best_param = self.optim.best.x
        self.tuner.unflatten_params(_as_float_list(best_param.solution))
        res = self.tuner.evaluate_fn()
        return res

    def get_diverse_top_parameters(self, num_top):
        df = self.archive.as_pandas()
        objectives = list(df.objective_batch())
        solutions = _as_2d_float_list(df.solution_batch())
        # store to allow restoring on next step
        if len(solutions) > num_top:
            # Select top-k solutions by objective without NumPy
            indices = sorted(range(len(objectives)), key=lambda i: float(objectives[i]), reverse=True)[:num_top]
            solutions = [_as_float_list(solutions[i]) for i in indices]

        return [self.tuner.unflatten_params(_as_float_list(x), apply=False) for x in solutions]
