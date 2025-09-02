# -*- coding: utf-8 -*-
# ================== 자동 Table & Cox 분석기 (penalizer Auto-CV 포함) ==================
# 필요 패키지: pandas, numpy, scipy, lifelines, openpyxl, xlrd, streamlit
# 설치: pip install -U pandas numpy scipy lifelines openpyxl xlrd streamlit

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import io
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# ----- 페이지 설정 -----
st.set_page_config(page_title="자동 논문 Table", layout="wide")

# ----- 간단 비밀번호 보호 (기본: CRCR 또는 st.secrets['APP_PASSWORD']) -----
def _check_password():
    def _password_entered():
        target = st.secrets.get("APP_PASSWORD", "CRCR")
        if st.session_state.get("_password_input", "") == str(target):
            st.session_state["_pw_ok"] = True
            st.session_state.pop("_password_input", None)
        else:
            st.session_state["_pw_ok"] = False

    if st.session_state.get("_pw_ok", False):
        return True

    st.sidebar.subheader("🔐 Access")
    st.sidebar.text_input("Password", type="password", key="_password_input", on_change=_password_entered)
    if st.session_state.get("_pw_ok") is False:
        st.sidebar.error("비밀번호가 올바르지 않습니다.")
    st.stop()

# 비밀번호 체크
_check_password()

st.title("자동 Table 생성기")

# -------------------- [공통 유틸] --------------------
def ensure_binary_outcome(col, positives:set, negatives:set):
    def _m(x):
        if x in positives: return 1
        if x in negatives: return 0
        return np.nan
    return col.apply(_m).astype(float)

def compute_vif(X):
    cols = [c for c in X.columns if c != 'const']
    if not cols:
        return pd.DataFrame(columns=['variable','VIF'])
    Xv = X[cols].astype(float)
    out = []
    for i, c in enumerate(cols):
        try:
            vif = variance_inflation_factor(Xv.values, i)
        except Exception:
            vif = np.nan
        out.append((c, float(vif) if np.isfinite(vif) else np.nan))
    return pd.DataFrame(out, columns=['variable','VIF'])

def hosmer_lemeshow_test(y_true, p_pred, g=10):
    df_hl = pd.DataFrame({'y':y_true, 'p':p_pred}).dropna()
    if df_hl['p'].nunique() < g:
        g = max(5, df_hl['p'].nunique())
    df_hl['bin'] = pd.qcut(df_hl['p'], q=g, duplicates='drop')
    grp = df_hl.groupby('bin').agg(obs=('y','sum'), exp=('p','sum'), n=('y','size')).reset_index()
    grp['pi'] = grp['exp']/grp['n']
    grp['var'] = grp['n'] * grp['pi'] * (1 - grp['pi'])
    grp = grp[grp['var'] > 0]
    chi2 = float(np.sum((grp['obs'] - grp['exp'])**2 / grp['var']))
    df = max(len(grp) - 2, 1)
    pval = 1 - stats.chi2.cdf(chi2, df)
    return chi2, df, float(pval)

def class_weights_balanced(y: pd.Series):
    """ Sklearn 'balanced' 방식을 모방한 빈도 가중치(합=샘플수로 정규화 안함) """
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    if n1 == 0 or n0 == 0:
        return pd.Series(np.ones(n), index=y.index)
    w1 = n / (2*n1); w0 = n / (2*n0)
    return y.map({1:w1, 0:w0})


def format_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p >= 0.999:
        return "p > 0.99"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def is_continuous(series, threshold=20):
    try:
        return (series.dtype.kind in "fi") and (series.nunique(dropna=True) > threshold)
    except Exception:
        return False

def ordered_levels(series):
    vals = pd.Series(series.dropna().unique()).tolist()
    numeric = []
    non = []
    for v in vals:
        try:
            numeric.append((float(str(v)), v))
        except Exception:
            non.append(str(v))
    if len(numeric) == len(vals) and len(vals) > 0:
        numeric.sort(key=lambda x: x[0])
        return [v for _, v in numeric]
    return sorted([str(v) for v in vals], key=lambda x: x)

def make_dummies(df_in, var, levels):
    # "변수=수준" 이름으로 더미 생성 (drop_first → 첫 레벨이 Reference)
    cat = pd.Categorical(df_in[var].astype(str),
                         categories=[str(x) for x in levels],
                         ordered=True)
    dmy = pd.get_dummies(cat, prefix=var, prefix_sep="=", drop_first=True, dtype=float)
    dmy.index = df_in.index
    return dmy

def dummy_colname(var, level):
    return f"{var}={str(level)}"

def drop_constant_cols(X):
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    return X[keep]

def drop_constant_predictors(X, time_col, event_col):  # === NEW: CV용 (time/event는 항상 유지)
    pred_cols = [c for c in X.columns if c not in [time_col, event_col]]
    keep = [c for c in pred_cols if X[c].nunique(dropna=True) > 1]
    return X[[time_col, event_col] + keep]

def clean_time(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def ensure_binary_event(col, events, censored):
    def _map(x):
        if x in events: return 1
        if x in censored: return 0
        return np.nan
    return col.apply(_map).astype(float)

# === NEW: penalizer를 CV로 선택 (C-index 최대화) ===
def select_penalizer_by_cv(X_all, time_col, event_col,
                           grid=(0.0, 0.01, 0.05, 0.1, 0.2, 0.5),
                           k=5, seed=42):
    """
    X_all: duration, event, predictors를 모두 포함한 데이터프레임 (dropna/상수열 제거된 상태 권장)
    반환: best_penalizer(or None), {penalizer: mean_cindex}
    """
    if X_all.shape[0] < k + 2 or X_all[event_col].sum() < k:  # 너무 작은 경우 방어
        return None, {}

    idx = X_all.index.to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)

    scores = {}
    for pen in grid:
        cv_scores = []
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            train = X_all.loc[train_idx].copy()
            test  = X_all.loc[test_idx].copy()

            # 학습셋에서 상수 predictor 제거, 열 일치 맞추기
            train = drop_constant_predictors(train, time_col, event_col)
            test  = test[train.columns]  # 같은 열 순서/구성 유지

            # 유효성 체크
            if train[event_col].sum() < 2 or test[event_col].sum() < 1:
                continue
            if train.shape[1] <= 2 or train.shape[0] < 5:
                continue

            try:
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(train, duration_col=time_col, event_col=event_col)
                # lifelines의 점수: concordance_index
                s = cph.score(test, scoring_method="concordance_index")
                s = float(s)
                if np.isfinite(s):
                    cv_scores.append(s)
            except Exception:
                continue

        if cv_scores:
            scores[pen] = float(np.mean(cv_scores))

    if not scores:
        return None, {}

    # 최고 C-index, 동점이면 더 작은 penalizer 선택
    best_pen = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return best_pen, scores

# -------------------- [파일 업로드] --------------------
uploaded_file = st.file_uploader("엑셀/CSV 업로드", type=['xls', 'xlsx', 'csv'])

df = None
sheetname = None

if uploaded_file:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        sheetname = xls.sheet_names[0] if len(xls.sheet_names) == 1 else st.selectbox("시트 선택", xls.sheet_names, index=0)
        df = pd.read_excel(xls, sheet_name=sheetname, engine="openpyxl")
    elif name.endswith(".xls"):
        xls = pd.ExcelFile(uploaded_file, engine="xlrd")
        sheetname = xls.sheet_names[0] if len(xls.sheet_names) == 1 else st.selectbox("시트 선택", xls.sheet_names, index=0)
        df = pd.read_excel(xls, sheet_name=sheetname, engine="xlrd")
    else:
        st.error("지원하지 않는 파일 형식입니다. csv/xls/xlsx만 업로드하세요.")
        st.stop()

    # 컬럼 공백/개행 제거 & 중복 방지
    df.columns = pd.Index([str(c).strip() for c in df.columns]).map(lambda x: x.replace("\\n", " ").strip())
    st.success(f"시트명: {sheetname if sheetname else ''}, 데이터 shape: {df.shape}")
    st.dataframe(df.head())
    st.session_state['df'] = df

# -------------------- [Table1: 서브 함수] --------------------
def calc_subgroup_p(valid, group_col, var, val, group_values):
    table = []
    for g in group_values:
        sub = valid[valid[group_col] == g][var]
        count_val = (sub == val).sum()
        count_else = (sub != val).sum()
        table.append([count_val, count_else])
    table = np.array(table)
    try:
        if table.shape == (2, 2):
            _, p = stats.fisher_exact(table)
        else:
            _, p, _, _ = stats.chi2_contingency(table)
    except Exception:
        p = np.nan
    return format_p(p)

def analyze_table1_display(df, group_col, value_map, threshold=20):
    result_rows = []
    group_values = list(value_map.keys())
    group_names = list(value_map.values())
    group_n = {g: (df[group_col] == g).sum() for g in group_values}

    for var in df.columns:
        if var == group_col: 
            continue
        valid = df[df[group_col].isin(group_values)]
        if valid[var].dropna().empty:
            continue

        # 연속형
        if is_continuous(valid[var], threshold=threshold):
            row = {'Characteristic': var}
            for g, g_name in zip(group_values, group_names):
                sub = valid[valid[group_col] == g][var].dropna()
                n = sub.shape[0]
                if n > 0:
                    med, q1, q3 = sub.median(), sub.quantile(0.25), sub.quantile(0.75)
                    mean, std = sub.mean(), sub.std()
                    row[f"{g_name} (n={group_n[g]})"] = f"{med:.1f} [{q1:.1f}-{q3:.1f}]; {mean:.1f}±{std:.1f}"
                else:
                    row[f"{g_name} (n={group_n[g]})"] = "NA"
            # 검정
            p = np.nan; test_str = ""
            try:
                arr = [valid[valid[group_col] == g][var].dropna() for g in group_values]
                normal_flags = []
                for vals in arr:
                    if len(vals) >= 3:
                        try:
                            p_norm = stats.shapiro(vals)[1]
                        except Exception:
                            p_norm = 0
                        normal_flags.append(p_norm > 0.05)
                    else:
                        normal_flags.append(False)
                if all(normal_flags):
                    if len(arr) == 2:
                        _, p = stats.ttest_ind(arr[0], arr[1], nan_policy='omit'); test_str = "t-test"
                    elif len(arr) > 2:
                        _, p = stats.f_oneway(*arr); test_str = "ANOVA"
                else:
                    if len(arr) == 2:
                        _, p = stats.mannwhitneyu(arr[0], arr[1], alternative='two-sided'); test_str = "Mann-Whitney U"
                    elif len(arr) > 2:
                        _, p = stats.kruskal(*arr); test_str = "Kruskal-Wallis"
            except Exception:
                p = np.nan; test_str = "NA"
            row['Test'] = test_str
            row['p value'] = format_p(p)
            row['sub_p'] = ""
            result_rows.append(row)
            continue

        # 범주형
        vlist = pd.Series(valid[var].dropna().unique())
        if len(vlist) >= threshold:
            row = {'Characteristic': var}
            for g, g_name in zip(group_values, group_names):
                sub = valid[valid[group_col] == g][var]
                vc = sub.value_counts()
                valstr = "; ".join([f"{idx}={cnt}({(cnt/group_n[g]*100):.0f}%)" for idx, cnt in vc.items()]) if group_n[g] > 0 else "NA"
                row[f"{g_name} (n={group_n[g]})"] = valstr if len(vc) > 0 else "NA"
            # 전체 p
            p = np.nan; test_str = ""
            try:
                ct = pd.crosstab(valid[group_col], valid[var])
                if ct.shape == (2, 2):
                    _, p = stats.fisher_exact(ct); test_str = "Fisher"
                else:
                    _, p, _, _ = stats.chi2_contingency(ct); test_str = "Chi-square"
            except Exception:
                p = np.nan; test_str = "NA"
            row['Test'] = test_str
            row['p value'] = format_p(p)
            row['sub_p'] = ""
            result_rows.append(row)
        else:
            # 헤더 행
            p = np.nan; test_str = ""
            try:
                ct = pd.crosstab(valid[group_col], valid[var])
                if ct.shape == (2, 2):
                    _, p = stats.fisher_exact(ct); test_str = "Fisher"
                else:
                    _, p, _, _ = stats.chi2_contingency(ct); test_str = "Chi-square"
            except Exception:
                p = np.nan; test_str = "NA"
            first_row = {'Characteristic': var}
            for g, g_name in zip(group_values, group_names):
                first_row[f"{g_name} (n={group_n[g]})"] = ""
            first_row['Test'] = test_str
            first_row['p value'] = format_p(p)
            first_row['sub_p'] = ""
            result_rows.append(first_row)

            for val in vlist:
                row = {'Characteristic': f"  {val}"}
                for g, g_name in zip(group_values, group_names):
                    cnt = valid[(valid[group_col] == g) & (valid[var] == val)].shape[0]
                    percent = (cnt/group_n[g]*100) if group_n[g] > 0 else 0
                    row[f"{g_name} (n={group_n[g]})"] = f"{cnt}({percent:.0f}%)"
                row['Test'] = ""
                row['p value'] = ""
                row['sub_p'] = calc_subgroup_p(valid, group_col, var, val, group_values)
                result_rows.append(row)

    return pd.DataFrame(result_rows)

# -------------------- [UI: Tab 구성] --------------------
if 'df' in st.session_state:
    df = st.session_state['df']

if 'df' not in locals():
    df = None

if df is not None:
    tab1, tab2, tab3 = st.tabs([
    "📊 Table1 자동화",
    "🟦 Cox 회귀분석 (Univariate/Multivariate)",
    "🟥 Logistic Regression (Binary Outcome)"
])


    # ===== TAB1 =====
    with tab1:
        st.header("2단계: Table 자동화 (값 직접 선택/라벨/행분리/요약 지원)")
        st.info(
            "📌 연속형/범주형 자동 분류: 고유값 20개 초과 → 연속형, 이하는 범주형.\\n"
            "결과가 상식과 다르면 직접 변수 타입을 확인하세요."
        )

        candidate_cols = list(df.columns)
        group_col = st.selectbox("분석할 그룹 변수명을 선택하세요", options=candidate_cols, key='group_col')
        value_map = {}

        if group_col and group_col in df.columns:
            unique_vals = list(df[group_col].dropna().unique())
            selected_vals = st.multiselect("분석할 값을 선택하세요", unique_vals, default=unique_vals[:2], key='group_values')
            if selected_vals:
                col1, col2 = st.columns([2,6])
                for val in selected_vals:
                    with col1:
                        st.write(f"값: {val}")
                    with col2:
                        label = st.text_input(f"해당 값의 표시 라벨", value=str(val), key=f'label_{val}')
                        value_map[val] = label

                if st.button("논문 Table1 생성", key='table1_generate'):
                    target_df = df.dropna(subset=[group_col])
                    result = analyze_table1_display(target_df, group_col, value_map, threshold=20)
                    st.dataframe(result)

                    output = io.BytesIO()
                    with pd.ExcelWriter(output) as writer:
                        result.to_excel(writer, index=False)
                    st.download_button(
                        label="Table1 엑셀로 저장",
                        data=output.getvalue(),
                        file_name="논문용_Table1.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("분석할 값을 최소 1개 이상 선택해 주세요.")
        elif group_col:
            st.error("해당 변수명이 데이터에 존재하지 않습니다.")

    # ===== TAB2 =====
    with tab2:
        st.header("논문 Table: Factor / Subgroup / HR(95%CI) / p-value (Univariate & Multivariate)")

        time_col = st.selectbox("생존기간 변수(time)", df.columns, key="cox_time_col")
        event_col = st.selectbox("Event 변수(예: 0=생존, 1=사망 등)", df.columns, key="cox_event_col")

        temp_df = df.copy()
        if event_col:
            unique_events = list(df[event_col].dropna().unique())
            st.write(f"이 변수의 실제 값: {unique_events}")
            selected_event = st.multiselect("이벤트(사건) 값", unique_events, key='selected_event_val')
            selected_censored = st.multiselect("생존/관찰종결(censored) 값", unique_events, key='selected_censored_val')
            st.caption("※ 사건값과 검열값은 서로 겹치면 안 됩니다.")
            temp_df["__event_for_cox"] = ensure_binary_event(temp_df[event_col], set(selected_event), set(selected_censored))
        else:
            temp_df["__event_for_cox"] = np.nan

        candidate_vars = [c for c in df.columns if c not in [time_col, event_col]]
        variables = st.multiselect("분석 후보 변수 선택", candidate_vars, key="cox_variables")

        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            p_enter = st.number_input("다변량 포함 기준 p-enter (≤)", min_value=0.001, max_value=1.0, value=0.05, step=0.01)
        with c2:
            max_levels = st.number_input("범주형 판정 최대 고유값", min_value=2, max_value=50, value=10, step=1)
        with c3:
            auto_penal = st.checkbox("penalizer 자동 선택 (CV, C-index)", value=False)  # === NEW
        with c4:
            cv_k = st.number_input("CV folds (K)", min_value=3, max_value=10, value=5, step=1, disabled=not auto_penal)  # === NEW

        penal_col = st.columns(1)[0]
        penalizer = penal_col.number_input("penalizer (수렴 안정화)", min_value=0.0, max_value=5.0, value=0.1, step=0.1, disabled=auto_penal)

        def basic_clean(df_in, time_col):
            out = df_in.copy()
            out[time_col] = clean_time(out[time_col])
            out = out[out[time_col] > 0]
            out = out.replace([np.inf, -np.inf], np.nan)
            return out

        if st.button("분석 실행"):
            # 필수 검증
            if not selected_event or not selected_censored:
                st.error("사건값과 검열값을 각각 최소 1개 이상 선택하세요.")
                st.stop()
            if set(selected_event) & set(selected_censored):
                st.error("사건값과 검열값이 겹칩니다. 다시 선택하세요.")
                st.stop()

            temp_df2 = basic_clean(temp_df, time_col).dropna(subset=[time_col, "__event_for_cox"])
            n_events = int(temp_df2["__event_for_cox"].sum())
            n_total = temp_df2.shape[0]
            st.info(f"총 관측치: {n_total}, 이벤트 수: {n_events}")
            if n_events < 5:
                st.warning("이벤트 수가 <5로 매우 적습니다. 추정이 불안정하거나 모델이 실패할 수 있습니다.")

            # ---------- 1) Univariate ----------
            uni_sum_dict = {}
            uni_na_vars = []
            cat_info = {}

            for var in variables:
                try:
                    dat_raw = temp_df2[[time_col, "__event_for_cox", var]].copy()
                    dat_raw = dat_raw.dropna(subset=[var])
                    if dat_raw.empty:
                        uni_na_vars.append(var); continue

                    if (dat_raw[var].dtype == "object") or (dat_raw[var].nunique(dropna=True) <= max_levels):
                        lvls = ordered_levels(dat_raw[var])
                        cat_info[var] = {"levels": lvls, "ref": lvls[0]}
                        dmy = make_dummies(dat_raw, var, lvls)
                        dat = pd.concat([dat_raw[[time_col, "__event_for_cox"]], dmy], axis=1)
                    else:
                        cat_info[var] = {"levels": None, "ref": None}
                        dat = dat_raw[[time_col, "__event_for_cox", var]].copy()
                        dat[var] = pd.to_numeric(dat[var], errors="coerce")

                    dat = dat.dropna()
                    dat = drop_constant_cols(dat)
                    if (dat.shape[0] < 3) or (dat["__event_for_cox"].sum() < 1) or (dat.shape[1] <= 2):
                        uni_na_vars.append(var); continue

                    cph = CoxPHFitter(penalizer=penalizer)  # Univariate는 입력 penalizer 사용
                    cph.fit(dat, duration_col=time_col, event_col="__event_for_cox")
                    uni_sum_dict[var] = cph.summary.copy()
                except ConvergenceError:
                    uni_na_vars.append(var)
                except Exception:
                    uni_na_vars.append(var)

            # 변수선택
            univariate_pvals = {}
            for var, summ in uni_sum_dict.items():
                if cat_info[var]["levels"] is None:
                    if var in summ.index:
                        univariate_pvals[var] = float(summ.loc[var, "p"])
                else:
                    p_min = None
                    for _, row in summ.iterrows():
                        p = float(row["p"])
                        p_min = p if p_min is None else min(p_min, p)
                    if p_min is not None:
                        univariate_pvals[var] = p_min

            selected_vars = [v for v, p in univariate_pvals.items() if p <= p_enter]
            st.write(f"다변량 후보 변수(≤ {p_enter:.3f}): {selected_vars if selected_vars else '없음'}")

            # ---------- 2) Multivariate ----------
            multi_sum = None
            multi_na_vars = []
            chosen_penalizer = penalizer  # 기본값

            if len(selected_vars) >= 1:
                try:
                    dat_base = temp_df2[[time_col, "__event_for_cox"]].copy()
                    X_list = []
                    for var in selected_vars:
                        if cat_info.get(var, {}).get("levels") is None:
                            xi = pd.to_numeric(temp_df2[var], errors="coerce").to_frame(var)
                        else:
                            lvls = cat_info[var]["levels"]
                            xi = make_dummies(temp_df2[[var]], var, lvls)
                        X_list.append(xi)
                    X_all = pd.concat([dat_base] + X_list, axis=1).dropna()
                    X_all = drop_constant_predictors(X_all, time_col, "__event_for_cox")

                    # === NEW: Auto-CV로 penalizer 선택 ===
                    if auto_penal and X_all["__event_for_cox"].sum() >= cv_k:
                        pen_grid = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5)
                        best_pen, pen_scores = select_penalizer_by_cv(
                            X_all, time_col, "__event_for_cox",
                            grid=pen_grid, k=int(cv_k), seed=42
                        )
                        if best_pen is not None:
                            chosen_penalizer = float(best_pen)
                            st.success(f"Auto-CV 선택 penalizer = {chosen_penalizer} (평균 C-index 기준)")
                            st.caption(f"Grid 성능: { {k: round(v,4) for k,v in pen_scores.items()} }")
                        else:
                            st.warning("CV로 penalizer를 결정하지 못했습니다. 입력값을 사용합니다.")

                    if (X_all.shape[0] >= 3) and (X_all["__event_for_cox"].sum() >= 1) and (X_all.shape[1] > 2):
                        cph_multi = CoxPHFitter(penalizer=chosen_penalizer)
                        cph_multi.fit(X_all, duration_col=time_col, event_col="__event_for_cox")
                        multi_sum = cph_multi.summary.copy()
                    else:
                        multi_na_vars = selected_vars
                except ConvergenceError:
                    multi_na_vars = selected_vars
                except Exception:
                    multi_na_vars = selected_vars

            # ---------- 3) 출력 테이블 ----------
            rows = []
            for var in variables:
                rows.append({
                    "Factor": var, "Subgroup": "",
                    "Univariate analysis HR (95% CI)": "", "Univariate analysis p-Value": "",
                    "Multivariate analysis HR (95% CI)": "", "Multivariate analysis p-Value": ""
                })

                # 완전 실패
                if (var in uni_na_vars) and ((multi_sum is None) or (var in multi_na_vars)):
                    rows.append({
                        "Factor": "", "Subgroup": "(insufficient / skipped)",
                        "Univariate analysis HR (95% CI)": "NA", "Univariate analysis p-Value": "NA",
                        "Multivariate analysis HR (95% CI)": "NA", "Multivariate analysis p-Value": "NA"
                    })
                    continue

                # 범주형
                if cat_info.get(var, {}).get("levels") is not None:
                    lvls = cat_info[var]["levels"]; ref = cat_info[var]["ref"]
                    rows.append({
                        "Factor": "", "Subgroup": f"{ref} (Reference)",
                        "Univariate analysis HR (95% CI)": "Ref.", "Univariate analysis p-Value": "",
                        "Multivariate analysis HR (95% CI)": "Ref.", "Multivariate analysis p-Value": ""
                    })
                    for lv in lvls[1:]:
                        colname = dummy_colname(var, lv)
                        # Uni
                        if (var in uni_na_vars) or (var not in uni_sum_dict) or (colname not in uni_sum_dict[var].index):
                            hr_uni, p_uni = "NA", "NA"
                        else:
                            r = uni_sum_dict[var].loc[colname]
                            hr_uni = f"{r['exp(coef)']:.3f} ({r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f})"
                            p_uni = format_p(float(r['p']))
                        # Multi
                        if (multi_sum is None) or (var in multi_na_vars) or (colname not in (multi_sum.index if multi_sum is not None else [])):
                            hr_multi, p_multi = "NA", "NA"
                        else:
                            r = multi_sum.loc[colname]
                            hr_multi = f"{r['exp(coef)']:.3f} ({r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f})"
                            p_multi = format_p(float(r['p']))
                        rows.append({
                            "Factor": "", "Subgroup": str(lv),
                            "Univariate analysis HR (95% CI)": hr_uni, "Univariate analysis p-Value": p_uni,
                            "Multivariate analysis HR (95% CI)": hr_multi, "Multivariate analysis p-Value": p_multi
                        })

                # 연속형
                else:
                    if (var not in uni_sum_dict) or (var in uni_na_vars) or (var not in uni_sum_dict[var].index if var in uni_sum_dict else True):
                        hr_uni, p_uni = "NA", "NA"
                    else:
                        r = uni_sum_dict[var].loc[var]
                        hr_uni = f"{r['exp(coef)']:.3f} ({r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f})"
                        p_uni = format_p(float(r['p']))

                    if (multi_sum is None) or (var in multi_na_vars) or (var not in (multi_sum.index if multi_sum is not None else [])):
                        hr_multi, p_multi = "NA", "NA"
                    else:
                        r = multi_sum.loc[var]
                        hr_multi = f"{r['exp(coef)']:.3f} ({r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f})"
                        p_multi = format_p(float(r['p']))

                    rows.append({
                        "Factor": "", "Subgroup": "",
                        "Univariate analysis HR (95% CI)": hr_uni, "Univariate analysis p-Value": p_uni,
                        "Multivariate analysis HR (95% CI)": hr_multi, "Multivariate analysis p-Value": p_multi
                    })

            result_table = pd.DataFrame(rows)
            st.write("**논문 제출용 테이블 (Univariate/Multivariate 병렬, Reference, Factor/수준구조)**")
            if auto_penal and len(selected_vars) >= 1:
                st.caption(f"*다변량 최종 penalizer: {chosen_penalizer}*")
            st.dataframe(result_table)

            output = io.BytesIO()
            with pd.ExcelWriter(output) as writer:
                result_table.to_excel(writer, index=False)
            st.download_button(
                label="Cox 결과 엑셀로 저장",
                data=output.getvalue(),
                file_name="Cox_Regression_Results_Table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
#tab3
    with tab3:
        st.header("질병 위험인자 로지스틱 회귀 (SPSS 스타일)")

        # ---------- 1) 결과변수/사건·비사건 매핑 (사용자 선택) ----------
        y_col = st.selectbox("결과변수(이진) 선택", df.columns, key="logit_ycol")
        uniq_y = list(df[y_col].dropna().unique()) if y_col else []
        st.caption(f"해당 변수의 고유값: {uniq_y}")

        pos_vals = st.multiselect("사건(=1) 값 선택", options=uniq_y, key="logit_pos")
        neg_vals = st.multiselect("비사건(=0) 값 선택", options=uniq_y, key="logit_neg")
        st.caption("※ 사건과 비사건 값은 겹치면 안 됩니다.")

        # ---------- 2) 예측변수 / 옵션 ----------
        X_candidates = [c for c in df.columns if c != y_col]
        X_vars = st.multiselect("예측 변수(후보) 선택", X_candidates, key="logit_vars")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            method = st.selectbox("방법", ["Enter(일괄)", "Forward LR(p-enter)"], index=0)
        with c2:
            p_enter = st.number_input("p-enter (Forward)", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
        with c3:
            max_levels = st.number_input("범주형 판정 최대 고유값", min_value=2, max_value=50, value=10, step=1)
        with c4:
            standardize = st.checkbox("연속형 표준화(옵션)", value=False)

        thr = st.slider("분류 임계값(threshold)", 0.05, 0.95, 0.50, 0.01)

        # 클래스 불균형 자동 감지(초기 제안)
        st.divider()
        st.caption("클래스 불균형이 심하면 가중 로지스틱(GLM Binomial, freq_weights)을 함께 제시합니다.")
        auto_weight = st.checkbox("이벤트가 적으면 가중모델도 함께 보기(추천)", value=True)
        manual_weight = st.checkbox("수동으로 가중모델도 항상 함께 보기", value=False)

        # ---------- 실행 버튼 ----------
        if st.button("로지스틱 회귀 실행"):
            # 입력 검증
            if not y_col or not X_vars:
                st.error("결과변수와 예측변수를 선택하세요.")
                st.stop()
            if not pos_vals or not neg_vals:
                st.error("사건/비사건 값을 각각 최소 1개 이상 지정하세요.")
                st.stop()
            if set(pos_vals) & set(neg_vals):
                st.error("사건값과 비사건값이 겹칩니다. 다시 선택하세요.")
                st.stop()

            # 데이터 구성
            Z = df[[y_col] + X_vars].copy()
            Z['__y__'] = ensure_binary_outcome(Z[y_col], set(pos_vals), set(neg_vals))

            # 결측 안내 → 삭제 확인 흐름
            miss_mask = Z[['__y__'] + X_vars].isna().any(axis=1)
            miss_n = int(miss_mask.sum())
            if miss_n > 0 and not st.session_state.get('logit_missing_ok', False):
                st.warning(f"선택한 변수들에서 결측치 {miss_n}행이 발견되었습니다. 분석에서 제외할까요?")
                colA, colB = st.columns(2)
                with colA:
                    if st.button("네, 결측 행 삭제 후 진행"):
                        st.session_state['logit_missing_ok'] = True
                        st.experimental_rerun()
                with colB:
                    if st.button("아니오, 변수 선택/데이터 확인"):
                        st.stop()
                st.stop()

            # (사용자가 승인한 경우) 결측 삭제
            Z = Z.dropna(subset=['__y__'] + X_vars)

            # 디자인 매트릭스 생성 (기존 유틸 사용)
            X_list, cat_info = [], {}
            for var in X_vars:
                s = Z[var]
                # 범주형 판단: 고유값 ≤ max_levels 또는 dtype object
                if (s.dtype == 'object') or (s.nunique(dropna=True) <= max_levels):
                    lvls = ordered_levels(s)
                    cat_info[var] = {'levels': lvls, 'ref': lvls[0]}
                    dmy = make_dummies(Z[[var]], var, lvls)  # drop_first=True 내부 구현 가정
                    X_list.append(dmy.astype(float))
                else:
                    x = pd.to_numeric(s, errors='coerce').to_frame(var)
                    if standardize:
                        x[var] = (x[var] - x[var].mean())/x[var].std(ddof=0)
                    cat_info[var] = {'levels': None, 'ref': None}
                    X_list.append(x.astype(float))

            Xd = pd.concat(X_list, axis=1)
            Xd = add_constant(Xd, has_constant='add')
            Xd = Xd.loc[:, Xd.nunique(dropna=True) > 1].copy()
            if 'const' not in Xd.columns:
                Xd = add_constant(Xd, has_constant='add')

            y = Z['__y__'].astype(int)
            n = len(y); n1 = int(y.sum()); n0 = n - n1
            p = Xd.shape[1] - 1  # const 제외

            if min(n1, n0) < 10:
                st.warning("사건/비사건 표본이 적습니다(각군 <10). 추정이 불안정할 수 있습니다.")
            if p > 0 and n1 < (10*p):
                st.info(f"경고: EPV(사건수/예측변수수)≈{n1}/{p} < 10. 변수 축소/병합 권고")

            # 방법: Forward 선택(있으면), 최종 변수 집합 결정
            selected_cols = [c for c in Xd.columns if c != 'const']
            if method.startswith("Forward"):
                selected, remaining = [], selected_cols.copy()
                while True:
                    best_p, best_var = None, None
                    for var in remaining:
                        try:
                            cols = ['const'] + selected + [var]
                            m = sm.Logit(y, Xd[cols]).fit(disp=0)
                            pv = m.pvalues.get(var, np.nan)
                            if np.isfinite(pv) and (best_p is None or pv < best_p):
                                best_p, best_var = float(pv), var
                        except Exception:
                            continue
                    if best_var is not None and best_p is not None and best_p <= p_enter:
                        selected.append(best_var)
                        remaining.remove(best_var)
                    else:
                        break
                if selected:
                    selected_cols = selected
                else:
                    st.info("Forward 기준을 만족하는 변수가 없어 Enter(전체)로 대체합니다.")
            final_cols = ['const'] + selected_cols

            # ---------- (A) 무가중치 Logit ----------
            try:
                modelA = sm.Logit(y, Xd[final_cols]).fit(disp=0)
                okA = True
            except Exception as e:
                st.error(f"무가중치 Logit 적합 실패: {e}")
                okA = False

            # ---------- (B) 가중 GLM(Binomial) (필요시/사용자요청시) ----------
            show_weighted = False
            if manual_weight:
                show_weighted = True
            elif auto_weight and (min(n1, n0) <= max(10, 0.15*n)):
                # 이벤트가 매우 적은 편이면 제안
                show_weighted = True

            modelB = None; okB = False; weights = None
            if show_weighted:
                try:
                    weights = class_weights_balanced(y)
                    glm = sm.GLM(y, Xd[final_cols],
                                 family=sm.families.Binomial(),
                                 freq_weights=weights)
                    modelB = glm.fit()
                    okB = True
                except Exception as e:
                    st.warning(f"가중 GLM 적합 실패(무가중치 결과만 표시): {e}")
                    okB = False

            def sp_ss_table(result, cat_info, title):
                coefs = result.params
                ses   = result.bse
                zvals = coefs/ses
                wald  = zvals**2
                # 일부 GLM은 'pvalues' 이름/계산이 다를 수 있어 getattr 사용
                pvals = getattr(result, 'pvalues', None)
                if pvals is None:
                    pvals = pd.Series(index=coefs.index, dtype=float)
                OR    = np.exp(coefs)
                ci_lo = np.exp(coefs - 1.96*ses)
                ci_hi = np.exp(coefs + 1.96*ses)

                rows = []
                rows.append({"Variable": f"— {title} —", "Subgroup":"", "B":"","SE":"","Wald χ²":"","p-value":"","OR(Exp(B))":"","95% CI":""})
                for var in X_vars:
                    rows.append({"Variable": var, "Subgroup":"", "B":"","SE":"","Wald χ²":"","p-value":"","OR(Exp(B))":"","95% CI":""})
                    if cat_info[var]['levels'] is not None:
                        ref = cat_info[var]['ref']
                        rows.append({"Variable":"", "Subgroup":f"{ref} (Reference)", "B":"Ref.","SE":"","Wald χ²":"","p-value":"","OR(Exp(B))":"","95% CI":""})
                        for lv in cat_info[var]['levels'][1:]:
                            name = f"{var}={lv}"
                            if name in coefs.index:
                                rows.append({
                                    "Variable":"", "Subgroup":str(lv),
                                    "B":f"{coefs[name]:.4f}", "SE":f"{ses[name]:.4f}",
                                    "Wald χ²":f"{wald[name]:.3f}",
                                    "p-value":f"{pvals[name]:.4f}" if name in pvals.index else "",
                                    "OR(Exp(B))":f"{OR[name]:.3f}",
                                    "95% CI":f"{ci_lo[name]:.3f} ~ {ci_hi[name]:.3f}"
                                })
                            else:
                                rows.append({"Variable":"", "Subgroup":str(lv),
                                             "B":"NA","SE":"NA","Wald χ²":"NA","p-value":"NA",
                                             "OR(Exp(B))":"NA","95% CI":"NA"})
                    else:
                        if var in coefs.index:
                            rows.append({
                                "Variable":"", "Subgroup":"",
                                "B":f"{coefs[var]:.4f}", "SE":f"{ses[var]:.4f}",
                                "Wald χ²":f"{wald[var]:.3f}",
                                "p-value":f"{pvals[var]:.4f}" if var in pvals.index else "",
                                "OR(Exp(B))":f"{OR[var]:.3f}",
                                "95% CI":f"{ci_lo[var]:.3f} ~ {ci_hi[var]:.3f}"
                            })
                        else:
                            rows.append({"Variable":"", "Subgroup":"",
                                         "B":"NA","SE":"NA","Wald χ²":"NA","p-value":"NA",
                                         "OR(Exp(B))":"NA","95% CI":"NA"})
                if 'const' in coefs.index:
                    rows.append({"Variable":"(Constant)", "Subgroup":"",
                                 "B":f"{coefs['const']:.4f}", "SE":f"{ses['const']:.4f}",
                                 "Wald χ²":f"{wald['const']:.3f}",
                                 "p-value":f"{pvals['const']:.4f}" if 'const' in pvals.index else "",
                                 "OR(Exp(B))":f"{OR['const']:.3f}",
                                 "95% CI":f"{ci_lo['const']:.3f} ~ {ci_hi['const']:.3f}"})
                return pd.DataFrame(rows)

            # 결과 출력: 표 + 적합도/성능 + VIF + 다운로드
            excel_buffers = {}
            if okA:
                st.subheader("회귀계수표 — 무가중치(Logit)")
                tblA = sp_ss_table(modelA, cat_info, "Unweighted Logit")
                st.dataframe(tblA, use_container_width=True)

                # 적합도/성능
                llf = float(modelA.llf); llnull = float(modelA.llnull)
                mcfadden = 1 - (llf/llnull) if llnull != 0 else np.nan
                lr_stat = 2*(llf - llnull)
                lr_p = 1 - stats.chi2.cdf(lr_stat, df=len(final_cols)-1)
                phatA = modelA.predict(Xd[final_cols]).astype(float)
                aucA = roc_auc_score(y, phatA)
                y_predA = (phatA >= thr).astype(int)
                cmA = confusion_matrix(y, y_predA, labels=[1,0])
                TP, FN, FP, TN = cmA[0,0], cmA[0,1], cmA[1,0], cmA[1,1]
                acc = (TP+TN)/cmA.sum() if cmA.sum() else np.nan
                sens = TP/(TP+FN) if (TP+FN)>0 else np.nan
                spec = TN/(TN+FP) if (TN+FP)>0 else np.nan
                ppv = TP/(TP+FP) if (TP+FP)>0 else np.nan
                npv = TN/(TN+FN) if (TN+FN)>0 else np.nan
                hl_chi2, hl_df, hl_p = hosmer_lemeshow_test(y, phatA, g=10)

                st.markdown("**모델 요약/적합도 (무가중치)**")
                st.write(f"- n={n}, p={len(final_cols)-1}")
                st.write(f"- Log-Likelihood={llf:.3f}, Null LL={llnull:.3f}")
                st.write(f"- McFadden pseudo R²={mcfadden:.3f}")
                st.write(f"- LR test: χ²={lr_stat:.3f}, df={len(final_cols)-1}, p={lr_p:.4f}")
                st.write(f"- Hosmer–Lemeshow: χ²={hl_chi2:.3f}, df={hl_df}, p={hl_p:.4f}")

                st.markdown("**분류 성능 (무가중치)**")
                st.write(f"- AUC={aucA:.3f} (threshold={thr:.2f})")
                st.write(f"- Acc={acc:.3f}, Sens={sens:.3f}, Spec={spec:.3f}, PPV={ppv:.3f}, NPV={npv:.3f}")
                st.write(pd.DataFrame(cmA, index=['Pred=1/True=1','Pred=0/True=0'], columns=['True=1','True=0']))

                # VIF
                st.markdown("**다중공선성(VIF)**")
                vif_df = compute_vif(Xd[final_cols])
                st.dataframe(vif_df, use_container_width=True)

                # excel 저장(무가중치)
                outA = io.BytesIO()
                with pd.ExcelWriter(outA) as wr:
                    tblA.to_excel(wr, sheet_name="logit_unweighted", index=False)
                    pd.DataFrame({'y_true': y, 'p_hat': phatA, 'y_pred': y_predA}).to_excel(wr, sheet_name="pred_unweighted", index=False)
                    vif_df.to_excel(wr, sheet_name="VIF", index=False)
                excel_buffers['unweighted'] = outA.getvalue()

            if okB:
                st.subheader("회귀계수표 — 가중 GLM(Binomial, freq_weights)")
                if weights is not None:
                    st.caption(f"가중치 예: 사건(1) 평균≈{weights[y==1].mean():.2f}, 비사건(0) 평균≈{weights[y==0].mean():.2f}")

                # 표
                tblB = sp_ss_table(modelB, cat_info, "Weighted GLM (Binomial)")
                st.dataframe(tblB, use_container_width=True)

                # 성능/적합도(주의: GLM은 Logit과 요약 지표 정의가 달라질 수 있음)
                phatB = modelB.predict(Xd[final_cols]).astype(float)
                aucB = roc_auc_score(y, phatB)
                y_predB = (phatB >= thr).astype(int)
                cmB = confusion_matrix(y, y_predB, labels=[1,0])
                TP, FN, FP, TN = cmB[0,0], cmB[0,1], cmB[1,0], cmB[1,1]
                acc = (TP+TN)/cmB.sum() if cmB.sum() else np.nan
                sens = TP/(TP+FN) if (TP+FN)>0 else np.nan
                spec = TN/(TN+FP) if (TN+FP)>0 else np.nan
                ppv = TP/(TP+FP) if (TP+FP)>0 else np.nan
                npv = TN/(TN+FN) if (TN+FN)>0 else np.nan
                hl_chi2, hl_df, hl_p = hosmer_lemeshow_test(y, phatB, g=10)

                st.markdown("**모델 요약/적합도 (가중)**")
                st.write(f"- Hosmer–Lemeshow: χ²={hl_chi2:.3f}, df={hl_df}, p={hl_p:.4f}")
                st.markdown("**분류 성능 (가중)**")
                st.write(f"- AUC={aucB:.3f} (threshold={thr:.2f})")
                st.write(f"- Acc={acc:.3f}, Sens={sens:.3f}, Spec={spec:.3f}, PPV={ppv:.3f}, NPV={npv:.3f}")
                st.write(pd.DataFrame(cmB, index=['Pred=1/True=1','Pred=0/True=0'], columns=['True=1','True=0']))

                # excel 저장(가중)
                outB = io.BytesIO()
                with pd.ExcelWriter(outB) as wr:
                    tblB.to_excel(wr, sheet_name="logit_weighted_glm", index=False)
                    pd.DataFrame({'y_true': y, 'p_hat': phatB, 'y_pred': y_predB}).to_excel(wr, sheet_name="pred_weighted", index=False)
                excel_buffers['weighted'] = outB.getvalue()

            # 다운로드 버튼(있으면 노출)
            if 'unweighted' in excel_buffers:
                st.download_button("무가중치 결과 엑셀 다운로드",
                                   excel_buffers['unweighted'],
                                   file_name="Logistic_Unweighted.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            if 'weighted' in excel_buffers:
                st.download_button("가중(GLM) 결과 엑셀 다운로드",
                                   excel_buffers['weighted'],
                                   file_name="Logistic_Weighted_GLM.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# ================== 끝 ==================
