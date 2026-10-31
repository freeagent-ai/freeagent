"""M8 - Signing-success classifier.

Question: when a club pays >= 5M euro for a player in the summer window, will he become a
REGULAR in his first season, i.e. play at least 50% of the new club's league minutes?

    y = 1  if  minutes_for_new_club(season) / (90 * club_league_games(season)) >= 0.5

Every feature is computed from information available ON THE TRANSFER DATE T:
  * player : age, position, height, foot, market value before T and its 1-year trend
  * deal   : fee, fee / market value, fee relative to the club's biggest fees in the 3 years before T
  * form   : last season's league minutes share, goals+assists per 90, European minutes
  * fit    : last season's strength of the buying club (points per game), promoted or not,
             same league or not, how crowded the player's position is at the new club
             (minutes of incumbents still at the club, and rival signings made before T)
  * ours   : the M3 Bayesian plus-minus rating, fitted only on matches before 1 June of the
             signing year, and the minutes behind it (how certain the rating is)
Label and features never share a season: features use season Y-1 and earlier, the label
uses season Y.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config

POS_SLOTS = {"Goalkeeper": 1, "Defender": 4, "Midfield": 3, "Attack": 3}
SUMMER_MONTHS = (6, 7, 8, 9, 10)
NUMERIC = ["age", "height", "log_fee", "log_value", "fee_to_value", "value_trend",
           "fee_vs_club_max", "prev_share", "prev2_share", "prev_ga90", "prev_euro_minutes",
           "to_ppg_prev", "promoted", "same_league", "from_top5", "crowding", "top_incumbent",
           "rival_signings", "rating", "rating_minutes"]
# Tested on validation seasons and NOT kept (no gain); still reported as an ablation.
VALUE_FEATURES = ["pos_value_rank", "value_vs_pos_best", "value_vs_squad"]
CONFIDENCE_MARGIN = 0.10      # "confident" = p >= 0.60 or p <= 0.40, fixed before testing
CATEGORICAL = ["position", "foot", "to_league"]
RATING_FEATURES = ["rating", "rating_minutes"]


# --------------------------------------------------------------------------- season tables
def league_tables(tables: dict) -> dict:
    """Per club-season and player-club-season summaries from top-5 league matches."""
    g = tables["games"]
    lg = g[g.competition_id.isin(config.TOP_LEAGUES)].dropna(subset=["home_club_goals"])
    h = lg[["game_id", "season", "competition_id", "home_club_id", "home_club_goals", "away_club_goals"]]
    a = lg[["game_id", "season", "competition_id", "away_club_id", "away_club_goals", "home_club_goals"]]
    cols = ["game_id", "season", "league", "club_id", "gf", "ga"]
    side = pd.concat([h.set_axis(cols, axis=1), a.set_axis(cols, axis=1)])
    side["pts"] = np.select([side.gf > side.ga, side.gf == side.ga], [3, 1], 0)
    club = side.groupby(["club_id", "season"]).agg(league=("league", "first"), n_games=("game_id", "size"),
                                                   ppg=("pts", "mean")).reset_index()
    apps = tables["appearances"].merge(g[["game_id", "season", "competition_id"]].rename(
        columns={"competition_id": "comp"}), on="game_id")
    la = apps[apps.comp.isin(config.TOP_LEAGUES)]
    pcs = la.groupby(["player_id", "player_club_id", "season"]).agg(
        mins=("minutes_played", "sum"), ga=("goals", "sum"), ast=("assists", "sum")).reset_index()
    pcs = pcs.merge(club[["club_id", "season", "n_games"]], left_on=["player_club_id", "season"],
                    right_on=["club_id", "season"]).drop(columns="club_id")
    pcs["share"] = (pcs.mins / (90 * pcs.n_games)).clip(upper=1)
    euro = apps[apps.comp.isin(config.CUPS)].groupby(["player_id", "season"]).minutes_played.sum()
    return {"club": club, "pcs": pcs, "euro": euro.rename("euro_minutes").reset_index()}


# --------------------------------------------------------------------------- signings + label
def build_signings(tables: dict, lt: dict | None = None, min_fee: float = config.MIN_FEE_EUR,
                   threshold: float = 0.5, last_season: int = 2025) -> pd.DataFrame:
    """Summer signings (fee >= min_fee) by clubs playing a top-5 league that season, with label."""
    lt = lt or league_tables(tables)
    t = tables["transfers"].dropna(subset=["transfer_date"])
    s = t[(t.transfer_fee >= min_fee) & t.transfer_date.dt.month.isin(SUMMER_MONTHS)].copy()
    s["season"] = s.transfer_date.dt.year
    s = s[s.season <= last_season]
    club = lt["club"]
    s = s.merge(club[["club_id", "season", "league", "n_games"]].rename(
        columns={"club_id": "to_club_id", "league": "to_league"}), on=["to_club_id", "season"])
    m = lt["pcs"][["player_id", "player_club_id", "season", "mins"]]
    s = s.merge(m, left_on=["player_id", "to_club_id", "season"],
                right_on=["player_id", "player_club_id", "season"], how="left").drop(columns="player_club_id")
    s["share"] = (s.mins.fillna(0) / (90 * s.n_games)).clip(upper=1)
    s["regular"] = (s.share >= threshold).astype(int)
    s = s.drop_duplicates(["player_id", "season", "to_club_id"])
    keep = ["player_id", "player_name", "transfer_date", "season", "from_club_id", "to_club_id",
            "from_club_name", "to_club_name", "to_league", "transfer_fee", "share", "regular"]
    for c in keep:
        if c not in s:
            s[c] = np.nan            # name columns are optional (absent in the synthetic world)
    return s[keep].sort_values("transfer_date").reset_index(drop=True)


# --------------------------------------------------------------------------- features
def _value_at(vals: pd.DataFrame, s: pd.DataFrame, when: pd.Series) -> np.ndarray:
    """Latest market value strictly before `when` for each row of s."""
    if len(vals) == 0:
        return np.full(len(s), np.nan)
    q =pd.DataFrame({"player_id": s.player_id.to_numpy(), "when": when.to_numpy(), "i": np.arange(len(s))})
    q = q.dropna(subset=["when"]).sort_values("when")
    v = vals.dropna(subset=["date", "market_value_in_eur"]).sort_values("date")
    r = pd.merge_asof(q, v[["player_id", "date", "market_value_in_eur"]], left_on="when", right_on="date",
                      by="player_id", allow_exact_matches=False)
    out = np.full(len(s), np.nan)
    out[r.i.to_numpy()] = r.market_value_in_eur.to_numpy()
    return out


def build_features(s: pd.DataFrame, tables: dict, lt: dict | None = None,
                   ratings: dict[int, "pd.DataFrame"] | None = None) -> pd.DataFrame:
    """Add pre-transfer features to the signings table. `ratings[Y]` = M3 ratings fitted before Y-06-01."""
    lt = lt or league_tables(tables)
    s = s.copy()
    T = s.transfer_date
    prev = s.season - 1
    players = tables["players"].set_index("player_id")
    col = lambda c: players[c] if c in players else pd.Series(dtype=object)
    s["position"] = s.player_id.map(col("position")).fillna("Unknown")
    s["foot"] = s.player_id.map(col("foot")).fillna("Unknown")
    s["height"] = pd.to_numeric(s.player_id.map(col("height_in_cm")), errors="coerce")
    s["age"] = (T - s.player_id.map(players.date_of_birth)).dt.days / 365.25

    # deal
    vals = tables.get("player_valuations", pd.DataFrame(columns=["player_id", "date", "market_value_in_eur"]))
    v_now = _value_at(vals, s, T)
    v_1y = _value_at(vals, s, T - pd.Timedelta(days=365))
    s["log_fee"] = np.log10(s.transfer_fee)
    s["log_value"] = np.log10(np.where(v_now > 0, v_now, np.nan))
    s["fee_to_value"] = s.transfer_fee / np.where(v_now > 0, v_now, np.nan)
    s["value_trend"] = v_now / np.where(v_1y > 0, v_1y, np.nan)
    tr = tables["transfers"].dropna(subset=["transfer_date", "transfer_fee"])
    tr = tr[tr.transfer_fee > 0]
    by_club = {c: d[["transfer_date", "transfer_fee"]].to_numpy() for c, d in tr.groupby("to_club_id")}
    fmax = []
    for c, d, f in zip(s.to_club_id, T, s.transfer_fee):
        x = by_club.get(c)
        if x is None:
            fmax.append(np.nan); continue
        w = x[(x[:, 0] < d) & (x[:, 0] >= d - pd.Timedelta(days=3 * 365)), 1]
        fmax.append(f / w.max() if len(w) else np.nan)
    s["fee_vs_club_max"] = fmax

    # form last season (top-5 leagues only; NaN = played outside the top 5)
    pcs = lt["pcs"]
    tot = pcs.groupby(["player_id", "season"]).agg(mins=("mins", "sum"), g=("ga", "sum"), a=("ast", "sum"),
                                                   n_games=("n_games", "max")).reset_index()
    tot["share"] = (tot.mins / (90 * tot.n_games)).clip(upper=1)
    tot["ga90"] = np.where(tot.mins >= 450, 90 * (tot.g + tot.a) / tot.mins.clip(lower=1), np.nan)
    key = tot.set_index(["player_id", "season"])
    idx1 = pd.MultiIndex.from_arrays([s.player_id, prev])
    idx2 = pd.MultiIndex.from_arrays([s.player_id, prev - 1])
    s["prev_share"] = key.share.reindex(idx1).to_numpy()
    s["prev2_share"] = key.share.reindex(idx2).to_numpy()
    s["prev_ga90"] = key.ga90.reindex(idx1).to_numpy()
    euro = lt["euro"].set_index(["player_id", "season"]).euro_minutes
    s["prev_euro_minutes"] = euro.reindex(idx1).fillna(0).to_numpy()

    # club context
    club = lt["club"].set_index(["club_id", "season"])
    to_prev = pd.MultiIndex.from_arrays([s.to_club_id, prev])
    from_prev = pd.MultiIndex.from_arrays([s.from_club_id, prev])
    s["to_ppg_prev"] = club.ppg.reindex(to_prev).to_numpy()
    s["promoted"] = s.to_ppg_prev.isna().astype(int)
    from_league = club.league.reindex(from_prev).to_numpy()
    s["from_top5"] = pd.notna(from_league).astype(int)
    s["same_league"] = (from_league == s.to_league.to_numpy()).astype(int)

    # positional competition at the new club, known at T
    pos = col("position")
    p_prev = pcs[["player_id", "player_club_id", "season", "share"]].assign(pos=lambda d: d.player_id.map(pos))
    outs = tr[["player_id", "from_club_id", "transfer_date"]]
    sig_all = s[["player_id", "to_club_id", "transfer_date", "position"]]
    crowd, top, rivals, inc_rows = [], [], [], []
    p_groups = {k: d for k, d in p_prev.groupby(["player_club_id", "season"])}
    out_groups = {k: d for k, d in outs.groupby("from_club_id")}
    sig_groups = {k: d for k, d in sig_all.groupby("to_club_id")}
    for i, (pid, c, d, y, ps) in enumerate(zip(s.player_id, s.to_club_id, T, s.season, s.position)):
        inc = p_groups.get((c, y - 1))
        if inc is None:
            crowd.append(np.nan); top.append(np.nan)
        else:
            inc = inc[inc.player_id != pid]
            o = out_groups.get(c)
            if o is not None:
                gone = o[(o.transfer_date >= pd.Timestamp(f"{y - 1}-07-01")) & (o.transfer_date < d)].player_id
                inc = inc[~inc.player_id.isin(gone)]
            same = inc[inc.pos == ps]
            crowd.append(same.share.sum() / POS_SLOTS.get(ps, 3))
            top.append(same.share.max() if len(same) else 0.0)
            inc_rows.append(pd.DataFrame({"i": i, "player_id": inc.player_id.to_numpy(),
                                          "same_pos": (inc.pos == ps).to_numpy(), "share": inc.share.to_numpy()}))
        r = sig_groups.get(c)
        rivals.append(0 if r is None else int(((r.transfer_date < d) & (r.transfer_date >= pd.Timestamp(f"{y}-06-01"))
                                               & (r.position == ps) & (r.player_id != pid)).sum()))
    s["crowding"], s["top_incumbent"], s["rival_signings"] = crowd, top, rivals

    # pecking order by market value at T: is he more valuable than the incumbents he must displace?
    s["pos_value_rank"], s["value_vs_pos_best"], s["value_vs_squad"] = np.nan, np.nan, np.nan
    if inc_rows:
        inc = pd.concat(inc_rows, ignore_index=True)
        inc["value"] = _value_at(vals, inc, T.iloc[inc.i.to_numpy()].reset_index(drop=True))
        inc["own"] = v_now[inc.i.to_numpy()]
        same = inc[inc.same_pos & inc.value.notna()]
        g = same.assign(above=same.value > same.own).groupby("i")
        rank = g.above.mean()
        best = g.value.max()
        core = inc[(inc.share >= 0.2) & inc.value.notna()].groupby("i").value.median()
        s.loc[rank.index, "pos_value_rank"] = rank.to_numpy()
        s.loc[best.index, "value_vs_pos_best"] = v_now[best.index] / best.to_numpy()
        s.loc[core.index, "value_vs_squad"] = v_now[core.index] / core.to_numpy()

    # our Bayesian plus-minus rating (M3), fitted before 1 June of the signing year
    s["rating"], s["rating_minutes"] = np.nan, np.nan
    if ratings:
        for y, r in ratings.items():
            m = s.season == y
            s.loc[m, "rating"] = s.loc[m, "player_id"].map(r.rating).to_numpy()
            s.loc[m, "rating_minutes"] = s.loc[m, "player_id"].map(r.minutes).to_numpy()
    for c in CATEGORICAL:
        s[c] = s[c].astype("category")
    return s


def fit_ratings(games: pd.DataFrame, stints: pd.DataFrame, seasons, lam: float = 100.0,
                halflife_days: float = 1460.0) -> dict[int, pd.DataFrame]:
    """M3 ratings for each signing season Y, using only matches before Y-06-01."""
    from .impact import PlusMinus
    out = {}
    for y in seasons:
        try:
            pm = PlusMinus(lam=lam, halflife_days=halflife_days).fit(games, stints, f"{y}-06-01")
        except ValueError:
            continue
        out[y] = pd.DataFrame({"rating": pm.ratings, "minutes": pm.minutes})
    return out


# --------------------------------------------------------------------------- extra features (M8 round 2)
EXTRA = ["share_trend", "rating_gap", "rating_rank", "inc_mean_age", "club_euro",
         "window_prior_signings", "fee_vs_window_max"]


def add_extra_features(s: pd.DataFrame, tables: dict, lt: dict | None = None,
                       ratings: dict | None = None) -> pd.DataFrame:
    """Second-round features, all known on the transfer date T (same leakage rules as build_features).
      share_trend            last season's minutes share minus the season before
      rating_gap / rank      his M3 rating minus the best same-position incumbent's rating / share of incumbents rated above him
      inc_mean_age           mean age of the regular same-position incumbents still at the club on T
      club_euro              log(1 + European minutes of last season's squad)  (rotation pressure)
      window_prior_signings  other EUR 5M+ signings by the club earlier in the same window
      fee_vs_window_max      his fee / the biggest earlier fee of that window
    `s` must already contain the build_features columns (prev_share, prev2_share, rating, position)."""
    lt = lt or league_tables(tables)
    s = s.copy()
    players = tables["players"].set_index("player_id")
    pos = players["position"] if "position" in players else pd.Series(dtype=object)
    dob = players.date_of_birth
    pcs = lt["pcs"]
    euro = lt["euro"].set_index(["player_id", "season"]).euro_minutes
    tr = tables["transfers"].dropna(subset=["transfer_date"])
    outs = {c: d[["player_id", "transfer_date"]] for c, d in tr.groupby("from_club_id")}
    reg = {k: d for k, d in pcs[pcs.share >= 0.2].groupby(["player_club_id", "season"])}
    allp = {k: d for k, d in pcs.groupby(["player_club_id", "season"])}
    sg = {c: d[["player_id", "season", "transfer_date", "transfer_fee"]] for c, d in s.groupby("to_club_id")}
    gap, rank, age_m, ceuro, nprior, fmax = [], [], [], [], [], []
    for pid, c, T, y, ps, fee, rt in zip(s.player_id, s.to_club_id, s.transfer_date, s.season, s.position,
                                         s.transfer_fee, s.rating):
        g = r = a = np.nan
        inc = reg.get((c, y - 1))
        if inc is not None:
            inc = inc[inc.player_id != pid]
            o = outs.get(c)
            if o is not None:
                gone = o[(o.transfer_date >= pd.Timestamp(f"{y - 1}-07-01")) & (o.transfer_date < T)].player_id
                inc = inc[~inc.player_id.isin(gone)]
            same = inc[inc.player_id.map(pos).to_numpy() == ps]
            if len(same):
                born = dob.reindex(same.player_id)
                a = float(((T - born).dt.days / 365.25).mean())
                if ratings and y in ratings and pd.notna(rt):
                    rr = ratings[y].rating.reindex(same.player_id).dropna()
                    if len(rr):
                        g, r = float(rt - rr.max()), float((rr > rt).mean())
        gap.append(g); rank.append(r); age_m.append(a)
        ap = allp.get((c, y - 1))
        if ap is None:
            ceuro.append(np.nan)
        else:
            e = euro.reindex(pd.MultiIndex.from_arrays([ap.player_id, np.full(len(ap), y - 1)])).fillna(0)
            ceuro.append(float(np.log1p(e.sum())))
        x = sg[c]
        prior = x[(x.season == y) & (x.transfer_date < T) & (x.player_id != pid)]
        nprior.append(len(prior))
        fmax.append(fee / prior.transfer_fee.max() if len(prior) else np.nan)
    s["share_trend"] = s.prev_share - s.prev2_share
    s["rating_gap"], s["rating_rank"], s["inc_mean_age"] = gap, rank, age_m
    s["club_euro"], s["window_prior_signings"], s["fee_vs_window_max"] = ceuro, nprior, fmax
    return s


# --------------------------------------------------------------------------- models
def make_model(kind: str, features: list[str], seed: int = config.SEED, **kw):
    """'gb' = gradient boosting (ours), 'logit' = logistic regression, 'market' = logistic on price only."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    if kind == "gb":
        params = dict(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                      min_samples_leaf=30, categorical_features="from_dtype", random_state=seed)
        params.update(kw)
        return HistGradientBoostingClassifier(**params)
    num = [f for f in features if f not in CATEGORICAL]
    cat = [f for f in features if f in CATEGORICAL]
    num_pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    pre = ColumnTransformer([("num", num_pipe, num)] +
                            ([("cat", OneHotEncoder(handle_unknown="ignore"), cat)] if cat else []))
    return make_pipeline(pre, LogisticRegression(C=kw.get("C", 1.0), max_iter=2000))


def make_baseline(kind: str, features: list[str], seed: int = config.SEED):
    """Model families used in the published transfer-success paper (random forest, AdaBoost, naive Bayes),
    re-implemented on OUR features so they can be compared under OUR protocol."""
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    num = [f for f in features if f not in CATEGORICAL]
    cat = [f for f in features if f in CATEGORICAL]
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median", add_indicator=True), num)] +
                            ([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)] if cat else []))
    clf = {"rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1),
           "ada": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=seed),
           "nb": GaussianNB()}[kind]
    return make_pipeline(pre, clf)


def make_panel_model(kind: str, features: list[str], seed: int = config.SEED, **kw):
    """Standard model families for the head-to-head comparison (all see the same features and preprocessing).
    kinds: rf, et, ada, nb, knn, svm, mlp, xgb (xgb needs the optional xgboost package)."""
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    num = [f for f in features if f not in CATEGORICAL]
    cat = [f for f in features if f in CATEGORICAL]
    num_pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    pre = ColumnTransformer([("num", num_pipe, num)] +
                            ([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)] if cat else []))
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=kw.get("leaf", 5), random_state=seed, n_jobs=-1)
    elif kind == "et":
        clf = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=kw.get("leaf", 5), random_state=seed, n_jobs=-1)
    elif kind == "ada":
        clf = AdaBoostClassifier(n_estimators=kw.get("n", 200), learning_rate=kw.get("lr", 0.5), random_state=seed)
    elif kind == "nb":
        clf = GaussianNB()
    elif kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=kw.get("k", 50))
    elif kind == "svm":
        clf = CalibratedClassifierCV(SVC(C=kw.get("C", 1.0)), method="sigmoid", cv=5, ensemble=False)
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=kw.get("hidden", (32,)), alpha=kw.get("alpha", 1e-2),
                            max_iter=600, random_state=seed)
    elif kind == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=kw.get("n", 200), max_depth=kw.get("depth", 3), learning_rate=0.05,
                            subsample=0.8, eval_metric="logloss", random_state=seed, n_jobs=-1)
    else:
        raise ValueError(kind)
    return make_pipeline(pre, clf)


class Ensemble:
    """Average of logistic regression and gradient boosting probabilities (ours).
    Logistic is stable on small data; boosting captures interactions such as age x position."""
    def __init__(self, features, C=0.03, **gb):
        self.models = [make_model("logit", features, C=C), make_model("gb", features, **gb)]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


class Majority:
    """Baseline: always predict the training base rate."""
    def fit(self, X, y):
        self.p = float(np.mean(y)); return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, balanced_accuracy_score
    pred = (p >= 0.5).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 and np.std(p) > 0 else 0.5
    return {"accuracy": float(np.mean(pred == y)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "auc": float(auc), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)), "n": int(len(y))}


def permutation_importance(model, X: pd.DataFrame, y, n_repeats: int = 20, seed: int = config.SEED) -> pd.Series:
    """Increase in log loss when one feature's values are shuffled (bigger = more important)."""
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = log_loss(y, model.predict_proba(X)[:, 1], labels=[0, 1])
    out = {}
    for c in X.columns:
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = X[c].to_numpy()[rng.permutation(len(X))]
            inc.append(log_loss(y, model.predict_proba(Xp)[:, 1], labels=[0, 1]) - base)
        out[c] = float(np.mean(inc))
    return pd.Series(out).sort_values(ascending=False)


def walk_forward(df: pd.DataFrame, model_fn, features: list[str], test_seasons, min_train_season: int = 2015):
    """For each test season Y: train on signings from seasons < Y, predict season Y."""
    out = []
    for y in test_seasons:
        tr = df[(df.season < y) & (df.season >= min_train_season)]
        te = df[df.season == y]
        if len(te) == 0:
            continue
        m = model_fn().fit(tr[features], tr.regular)
        out.append(te.assign(p=m.predict_proba(te[features])[:, 1]))
    return pd.concat(out)
