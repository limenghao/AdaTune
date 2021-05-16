import multiprocessing
import logging
import time
import numpy as np

from .model_based_tuner import CostModel, FeatureCache
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from .. import feature

logger = logging.getLogger('autotvm')

class RFModel(CostModel):
    def __init__(self, task, fea_type="itervar",num_threads=None, log_interval=25, upper_model=None):
        super().__init__()
        self.task = task
        self.target = task.target
        self.space = task.config_space
        
        self.prior = RandomForestRegressor(n_estimators=10, random_state=2, max_features=10)
        self.fea_type = fea_type
        self.num_threads = num_threads
        self.log_interval = log_interval
        if fea_type == 'itervar':
            self.feature_extract_func = _extract_itervar_feature_index
        elif fea_type == 'knob':
            self.feature_extract_func = _extract_knob_feature_index
        elif fea_type == "simpleknob":
            self.feature_extract_func = _extract_simpleknob_feature_index
        elif fea_type == 'curve':
            self.feature_extract_func = _extract_curve_feature_index
        else:
            raise RuntimeError("Invalid feature type " + fea_type)

        # self.feature_cache = FeatureCache()
        self.best_flops = 0.0
        if upper_model:  # share a same feature cache with upper model
            self.feature_cache = upper_model.feature_cache
        else:
            self.feature_cache = FeatureCache()
        self.upper_model = upper_model
        self.pool = None
        self._reset_pool(self.space, self.target, self.task)
        
        
    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""

        if self.upper_model:  # base model will reuse upper model's pool,
            self.upper_model._reset_pool(space, target, task)
            return

        self._close_pool()

        # use global variable to pass common arguments
        global _extract_space, _extract_target, _extract_task
        _extract_space = space
        _extract_target = target
        _extract_task = task
        self.pool = multiprocessing.Pool(self.num_threads)

    def _close_pool(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _get_pool(self):
        if self.upper_model:
            return self.upper_model._get_pool()
        return self.pool
        
    def _expected_imporvement(self, x_to_predict):
        feas = self._get_feature(x_to_predict)
        preds = np.array([tree.predict(feas) for tree in self.prior]).T
        eis = []
        variances = []
        for pred in preds:
            mu = np.mean(pred)
            sigma = pred.std()
            # print("mu: %f, sigma: %f" % (mu, sigma))
            best_flops = self.best_flops
            variances.append(sigma)
            with np.errstate(divide='ignore'):
                Z = (mu - best_flops) / sigma
                ei = (mu - best_flops) * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] == max(0.0, mu-best_flops)
            eis.append(ei)
        # print("return eis: " + str(eis))
        mean_of_variance = sum(variances)/len(variances)
        return np.array(eis), mean_of_variance
    
    def _get_feature(self, indexes):
        """get features for indexes, run extraction if we do not have cache for them"""
        # free feature cache
        if self.feature_cache.size(self.fea_type) >= 100000:
            self.feature_cache.clear(self.fea_type)

        fea_cache = self.feature_cache.get(self.fea_type)

        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in fea_cache]

        if need_extract:
            pool = self._get_pool()
            feas = pool.map(self.feature_extract_func, need_extract)
            for i, fea in zip(need_extract, feas):
                fea_cache[i] = fea

        feature_len = None
        for idx in indexes:
            if fea_cache[idx] is not None:
                feature_len = fea_cache[idx].shape[-1]
                break

        ret = np.empty((len(indexes), feature_len), dtype=np.float32)
        for i, ii in enumerate(indexes):
            t = fea_cache[ii]
            ret[i, :] = t if t is not None else 0
        return ret
    
    def fit(self, xs, ys, plan_size):
        """Fit to training data

        Parameters
        ----------
        xs: Array of int
            indexes of configs in the config space
        ys: Array of float
            The speed (flop, float number operations per second)
        plan_size: int
            The plan size of tuner
        """
        # here, xs is a list of config_index
        # transfer into corresbonding x_list of fea_type
        x_list = self._get_feature(xs)
        self.best_flops = max(ys)
        # print(self.best_flops)
        self.prior.fit(x_list, ys)

    def fit_log(self, records, plan_size):
        """Fit training data from log.

        Parameters
        ----------
        records: Array of Tuple(MeasureInput, MeasureResult)
            The tuning records
        plan_size: int
            The plan size of tuner
        """
        raise NotImplementedError()

    def predict(self, xs, output_margin=False):
        """Predict the speed of configs

        Parameters
        ----------
        xs: Array of int
            The indexes of configs to predict
        output_margin: bool, optional
            Whether output the untransformed margin.
            When a model is used as base model, it should output untransformed margin

        Returns
        -------
        preds: Array of float
            The prediction
        """
        predicts, variance = self._expected_imporvement(xs)
        return predicts
    
    def load_basemodel(self, base_model):
        self.base_model = base_model
        self.base_model._close_pool()
        self.base_model.upper_model = self

    def spawn_base_model(self):
        return RFModel(self.task, self.fea_type, self.loss_type,
                       self.num_threads, self.log_interval, self)

    def __del__(self):
        self._close_pool()
        
_extract_space = None
_extract_target = None
_extract_task = None

def _extract_itervar_feature_index(index):
    """extract iteration var feature for an index in extract_space"""
    try:
        config = _extract_space.get(index)
        with _extract_target:
            sch, args = _extract_task.instantiate(config)
        fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)
        fea = np.concatenate((fea, list(config.get_other_option().values())))
        return fea
    except Exception:  # pylint: disable=broad-except
        return None

def _extract_itervar_feature_log(arg):
    """extract iteration var feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        with inp.target:
            sch, args = inp.task.instantiate(config)
        fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)
        x = np.concatenate((fea, list(config.get_other_option().values())))

        if res.error_no == 0:
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None

def _extract_knob_feature_index(index):
    """extract knob feature for an index in extract_space"""
    try:
        config = _extract_space.get(index)
        return config.get_flatten_feature()
    except Exception:  # pylint: disable=broad-except
        return None

def _extract_knob_feature_log(arg):
    """extract knob feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        x = config.get_flatten_feature()

        if res.error_no == 0:
            with inp.target:  # necessary, for calculating flops of this task
                inp.task.instantiate(config)
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None
    
from .model_based_tuner import knob2point, point2knob
def _extract_simpleknob_feature_index(index):
    """take the knob as feature to train the model"""
    
    try:
        # config = _extract_space.get(index)
        # return config.get_flatten_feature()
        dims = [len(x) for x in _extract_space.space_map.values()]
        knob = point2knob(index, dims)
        return np.array(knob)
    except Exception:  # pylint: disable=broad-except
        return None

def _extract_simpleknob_feature_log(arg):
    """extract knob feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        # x = config.get_flatten_feature()
        dims = [len(x) for x in _extract_space.space_map.values()]
        x = point2knob(inp.config.index, dims)
        x = np.array(x)
        
        if res.error_no == 0:
            with inp.target:  # necessary, for calculating flops of this task
                inp.task.instantiate(config)
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None
    
def _extract_curve_feature_index(index):
    """extract sampled curve feature for an index in extract_space"""
    try:
        config = _extract_space.get(index)
        with _extract_target:
            sch, args = _extract_task.instantiate(config)
        fea = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=20)
        fea = np.concatenate((fea, list(config.get_other_option().values())))
        return np.array(fea)
    except Exception:  # pylint: disable=broad-except
        return None

def _extract_curve_feature_log(arg):
    """extract sampled curve feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        with inp.target:
            sch, args = inp.task.instantiate(config)
        fea = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=20)
        x = np.concatenate((fea, list(config.get_other_option().values())))

        if res.error_no == 0:
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None
        
from .model_based_tuner import ModelOptimizer
from .dynamic_model_based_tuner import ModelBasedTunerAda
from .sa_model_optimizer import SimulatedAnnealingOptimizer

class RFTuner(ModelBasedTunerAda):
    def __init__(self, task, plan_size=32,
                 feature_type='itervar', loss_type='rank', num_threads=None,
                 optimizer='sa', diversity_filter_ratio=None, log_interval=50, dynamic_ep=False):
        
        cost_model = RFModel(task, fea_type=feature_type)
        if optimizer == 'sa':
            optimizer = SimulatedAnnealingOptimizer(task, log_interval=log_interval, parallel_size=plan_size*2)
        else:
            assert isinstance(optimizer, ModelOptimizer), "Optimizer must be " \
                                                          "a supported name string" \
                                                          "or a ModelOptimizer object."
        super(RFTuner, self).__init__(task, cost_model, optimizer,
                                       plan_size, diversity_filter_ratio, dynamic_ep)
        
    def tune(self, *args, **kwargs):  # pylint: disable=arguments-differ
        super(RFTuner, self).tune(*args, **kwargs)
        # manually close pool to avoid multiprocessing issues
        self.cost_model._close_pool()
        