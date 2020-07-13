import os
import multiprocessing
import time
from datetime import datetime
from pprint import pprint
from pathlib import Path
import numpy as np
from numpy import nan, inf
from collections import defaultdict
import argparse

from .test_aspirin import test_aspirin
from .test_cockroaches import test_cockroaches
from .test_coin import test_coin
from .test_double_gaussian import test_double_gaussian
from .test_gaussian_log_density import test_gaussian_log_density
from .test_linear_regression import test_linear_regression
from .test_neal_funnel import test_neal_funnel
from .test_schools import test_schools
from .test_seeds import test_seeds
from .harness import Config

tests = [
    test_aspirin,
    test_cockroaches,
    test_coin,
    test_double_gaussian,
    test_gaussian_log_density,
    test_linear_regression,
    test_neal_funnel,
    test_schools,
    test_seeds,
]

name_map = {
    "coin": "coin",
    "double_gaussian": "double normal",
    "gaussian_log_density": "gaussian target",
    "neal_funnel": "reparameterization",
    "linear_regression": "linear regression",
    "aspirin": "aspirin",
    "cockroaches": "roaches",
    "schools": "8 schools",
    "seeds": "seeds",
}


def run(exp):
    f, i, config, logdir = exp
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name = f.__name__
    Path(logdir).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(logdir, f"{name}_{i}_{now}.log")
    try:
        results = f(config)
        with open(filename, "w") as file:
            pprint(results, file)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print(f"ERROR test {name}")
        raise


def run_all(config=Config(), logdir="logs", n_runs=5):
    experiments = [(t, i, config, logdir) for t in tests for i in range(n_runs)]
    n_cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_cpu)
    pool.map(run, experiments)
    pool.close()
    pool.join()


def flatten(d):
    res = {}
    for k, v in d.items():
        if set(v.keys()) == {"statistic", "pvalue"}:  # leaf values
            res[k] = v
        else:
            for kk, vv in flatten(v).items():
                res[f"{k}[{kk}]"] = vv
    return res


def get_log(f):
    with open(f, "r") as log_file:
        raw = log_file.read()
        log = eval(raw)
        res = {}
        res["timers"] = log["timers"]
        res["divergences"] = {
            "numpyro": flatten(log["divergences"]["numpyro"]["ks"]),
            "pyro": flatten(log["divergences"]["pyro"]["ks"]),
        }
        return res


def get_min(d):
    k = min(d, key=lambda key: d[key])
    return {"parameter": k, "pvalue": d[k]}


def crunch(dirname):
    summary = {}
    for x in name_map:
        results = {"timers": {}}
        divergences = {"pyro": {}, "numpyro": {}, "pyro_naive": {}, "numpyro_naive": {}}
        pyro = defaultdict(list)
        numpyro = defaultdict(list)
        pyro_naive = defaultdict(list)
        numpyro_naive = defaultdict(list)
        timers = defaultdict(list)
        files = [
            os.path.join(dirname, f)
            for f in os.listdir(dirname)
            if f.startswith(f"test_{x}")
        ]
        logs = [get_log(f) for f in files]
        for l in logs:
            print(l)
            for k, v in l["timers"].items():
                timers[k].append(v)
            for k, v in l["divergences"]["pyro"].items():
                pyro[k].append(v["pvalue"])
            for k, v in l["divergences"]["numpyro"].items():
                numpyro[k].append(v["pvalue"])
            for k, v in l["divergences"]["pyro_naive"].items():
                pyro_naive[k].append(v["pvalue"])
            for k, v in l["divergences"]["numpyro_naive"].items():
                numpyro_naive[k].append(v["pvalue"])
        for k, v in timers.items():
            results["timers"][k] = np.mean(v)
        for k, v in pyro.items():
            divergences["pyro"][k] = np.mean(v)
            results["Pyro_KS"] = get_min(divergences["pyro"])
        for k, v in numpyro.items():
            divergences["numpyro"][k] = np.mean(v)
            results["NumPyro_KS"] = get_min(divergences["numpyro"])
        for k, v in pyro_naive.items():
            divergences["pyro_naive"][k] = np.mean(v)
            results["Pyro_naive_KS"] = get_min(divergences["pyro_naive"])
        for k, v in numpyro_naive.items():
            divergences["numpyro_naive"][k] = np.mean(v)
            results["NumPyro_naive_KS"] = get_min(divergences["numpyro_naive"])
        summary[x] = results
    return summary


def to_tex(summary):
    print(r"\begin{tabular}{@{}lr@{ }lrr@{ }lrr@{ }l@{}}\\")
    print(
        r"&\multicolumn{2}{c}{\textsc{Stan}}& \multicolumn{3}{c}{\textsc{DeepStan/Pyro}} & \multicolumn{3}{c}{\textsc{DeepStan/NumPyro}}\\"
    )
    print(r"\cmidrule(lr){2-3}")
    print(r"\cmidrule(lr){4-6}")
    print(r"\cmidrule(lr){7-9}")
    print(
        r"&\multicolumn{2}{c}{time(s)} & time(s) & \multicolumn{2}{l}{KS p-value} & time(s) & \multicolumn{2}{l}{KS p-value}\\"
    )
    print(r"\toprule")
    for t in name_map:
        if t == "aspirin":
            print(r"\midrule")
        name = name_map[t]
        d = summary[t]
        t_stan = f"{d['timers']['Stan_Compilation']:,.0f} &+ {d['timers']['Stan_Runtime']:,.1f}"
        t_numpyro = f"{d['timers']['NumPyro_Compilation']:,.0f} + {d['timers']['NumPyro_Runtime']:,.0f}"
        nks_value = d["NumPyro_KS"]["pvalue"]
        nks_param = d["NumPyro_KS"]["parameter"]
        ks_numpyro = f"{nks_value:,.2f} & (\\stans|{nks_param}|)"
        if "Pyro_Runtime" in d["timers"]:
            t_pyro = f"{d['timers']['Pyro_Compilation']:,.0f} + {d['timers']['Pyro_Runtime']:,.0f}"
            pks_value = d["Pyro_KS"]["pvalue"]
            pks_param = d["Pyro_KS"]["parameter"]
            ks_pyro = f"{pks_value:,.2f} & (\\stans|{pks_param}|)"
        else:
            t_pyro = r"\_"
            ks_pyro = r"\_ & \_"
        print(
            f"{name} & {t_stan} & {t_pyro} & {ks_pyro} & {t_numpyro} & {ks_numpyro}\\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the output of NUTS for DeepStan (with Pyro and Numpyro) and Stan on the following experiments: coin, double normal, reparameterization, linear regression, aspirin, roaches, 8 schools, seeds"
    )

    parser.add_argument(
        "--logdir",
        action="store",
        dest="logdir",
        default="logs",
        help="Directory name to store the results",
    )

    parser.add_argument(
        "--iterations",
        action="store",
        dest="iterations",
        type=int,
        default=10000,
        help="Total number of iterations",
    )

    parser.add_argument(
        "--warmups",
        action="store",
        dest="warmups",
        type=int,
        default=1000,
        help="Number of warmup steps (included in iterations)",
    )

    parser.add_argument(
        "--thin",
        action="store",
        dest="thin",
        type=int,
        default=10,
        help="Thining factor",
    )

    parser.add_argument(
        "--runs",
        action="store",
        dest="n_runs",
        type=int,
        default=10,
        help="Number of run for each experiment",
    )

    parser.add_argument(
        "--no-run",
        action="store_true",
        dest="no_run",
        default=False,
        help="Analyse logdir without re-running the experiments",
    )

    parser.add_argument(
        "--to-tex",
        action="store_true",
        dest="tex",
        default=False,
        help="Print results in tex format",
    )

    args = parser.parse_args()

    config = Config(iterations=args.iterations, warmups=args.warmups, thin=args.thin)

    if not args.no_run:
        start = time.perf_counter()
        run_all(config, args.logdir, n_runs=args.n_runs)
        print(f"Total experiment time {time.perf_counter() - start}")
        print(f"Config: {config}")

    res = crunch(args.logdir)

    if args.tex:
        pprint(to_tex(res))
    else:
        pprint(res)
