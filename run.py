import pandas as pd
import numpy as np
import time
from nmfs.deepKLNMF import DeepNMFParams
from evaluations import SSNMFParam, NMFEvaluations
from dataclasses import dataclass, asdict
from typing import Optional, Literal, List, Dict
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
p = time.time()


@dataclass
class EvalRows:
    nmf: str
    ssnmf_model: Optional[int]
    init: str
    algo: Optional[str]
    layers: int
    rank: int
    score: float


# NOTE: using classification
df = pd.read_csv("./data/data.csv")
df["stab"] = df["Stability"].map({"unstable": 0, "stable": 1}).astype(int)
X = df.select_dtypes(include=[np.number]).drop(columns=["stab"]).copy()
y = df["stab"].copy()
# X = df.drop(
#     ["Material", "a", "b", "Composition", "Stability"], axis=1
# ).copy()
# y = df["a"].copy()
rank = 71
layers = 3
models = [3, 4, 5, 6]
algos = ["MUUP", "ADMM", "HALS", "FPGM", "ALSH"]
inits = ["random", "nndsvd", "nndsvda", "nndsvdar", "nnsvdlrc"]
nmfs = ["hier"]
nmfs = ["ssnmf", "deep", "multilayer", "fronorm", "beta", "hier"]
ranks = [i for i in range(2, rank + 1)]

deepparams = DeepNMFParams(
    L=layers,
    outerit=200,
    maxiter=200,
    rngseed=47,
    min_vol=False,
    rho=10,
    epsi=1e-10,
    display=False,
    accADMM=True,
    lam=[4, 2, 1],
)


def collect_evals(
    evals: List[EvalRows], nmf, layers, init, rank, score, ssnmf_model=None, algo=None
):
    evals.append(
        EvalRows(
            nmf=nmf,
            ssnmf_model=ssnmf_model,
            init=init,
            layers=layers,
            algo=algo,
            rank=rank,
            score=score[0],
        )
    )
    print(f"collect: nmf={nmf} | init={init} | rank={rank} | score={score} | model={ssnmf_model}")


evals: List[EvalRows] = []
for nmf in nmfs:
    for init in inits: 
        print('---------------> Next <---------------')
        nmf_evaluations = NMFEvaluations(
            df=df,
            X=X,
            y=y,
            task="classification",
            rank=rank,
            parallel=False,
            cross_validation=False,
            seed=47,
            deepparams=deepparams,
            ssnmfparams=SSNMFParam(),
            layers=layers,
        )
        if nmf == "fronorm":
            for algo in algos:
                scores = nmf_evaluations.evaluates(
                    nmf=nmf,
                    init=init,
                    evalutation_type="feature",
                    normalize_X="minmax",
                    normalize_init=None,
                    algo=algo,
                    ranks=ranks,
                    exports=None,
                )
                [
                    collect_evals(
                        evals,
                        nmf,
                        layers,
                        init,
                        ranks[i],
                        score,
                        ssnmf_model=None,
                        algo=algo,
                    )
                    for i, score in enumerate(scores)
                ]
        elif nmf == "ssnmf":
            for model_num in models:
                nmf_evaluations.ssnmfparams.model_num = model_num
                scores = nmf_evaluations.evaluates(
                    nmf=nmf,
                    init=init,
                    evalutation_type="feature",
                    normalize_X="minmax",
                    normalize_init=None,
                    algo=None,
                    ranks=ranks,
                    exports=None,
                )
                [
                    collect_evals(
                        evals,
                        nmf,
                        layers,
                        init,
                        ranks[i],
                        score,
                        ssnmf_model=model_num,
                        algo=None,
                    )
                    for i, score in enumerate(scores)
                ]
        else:
            scores = nmf_evaluations.evaluates(
                nmf=nmf,
                init=init,
                evalutation_type="feature",
                normalize_X="minmax",
                normalize_init=None,
                algo=None,
                ranks=ranks,
                exports=None,
            )
            [
                collect_evals(
                    evals,
                    nmf,
                    layers,
                    init,
                    ranks[i],
                    score,
                    ssnmf_model=None,
                    algo=None,
                )
                for i, score in enumerate(scores)
            ]

df_evals = pd.DataFrame([asdict(e) for e in evals])
df_evals.to_excel("./exports/evals_binary0.xlsx", index_label="i")
print(df_evals.head())
# print(evals)


# NOTE: using regression
