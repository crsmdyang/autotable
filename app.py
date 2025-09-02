# -*- coding: utf-8 -*-

# ================== 자동 Table & Cox 분석기 (penalizer Auto-CV 포함) ==================
# 필요 패키지: pandas, numpy, scipy, lifelines, openpyxl, xlrd, streamlit, statsmodels
# 설치: pip install -U pandas numpy scipy lifelines openpyxl xlrd streamlit statsmodels

import io
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError

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
    numeric, non = [], []
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
    cat = pd.Categorical(
        df_in[var].astype(str),
        categories=[str(x) for x in levels],
        ordered=True
    )
    dmy = pd.get_dummies(cat, prefix=var, prefix_sep="=", drop_first=True, dtype=float)
    dmy.index = df_in.index
    return dmy

def dummy_colname(var, level):
    return f"{var}={str(level)}"

def drop_constant_cols(X):
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    return X[keep]

def drop_constant_predictors(X, time_col, event_col):  # === CV용 (time/event는 항상 유지)
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
def select_penalizer_by_cv(
    X_all, time_col, event_col,
    grid=(0.0, 0.01, 0.05, 0.1, 0.2, 0.5),
    k=5, seed=42
):
    """
    X_all: duration, event, predictors를 모두 포함한 데이터프레임 (dropna/상수열 제거된 상태 권장)
    반환: best_penalizer(or None), {penalizer: mean_cindex}
    """
    if X_all.shape[0] < k + 2 or X_all[event_col].sum() < k:
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
    df.columns = pd.Index([str(c).strip() for c in df.columns]).map(lambda x: x.replace("\n", " ").strip())
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
    group_names  = list(value_map.values())
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
df = st.session_state.get('df')
if df is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Table1 자동화", "🟦 Cox 회귀분석 (Univariate/Multivariate)", "🟧 로지스틱 회귀분석 (Risk Factors)"])

    # ===== TAB1 =====
    with tab1:
        st.header("2단계: Table 자동화 (값 직접 선택/라벨/행분리/요약 지원)")
        st.info(
            "📌 연속형/범주형 자동 분류: 고유값 20개 초과 → 연속형, 이하는 범주형.\n"
            "결과가 상식과 다르면 직접 변수 타입을 확인하세요."
        )

        candidate_cols = list(df.columns)
        group_col = st.selectbox("분석할 그룹 변수명을 선택하세요", options=candidate_cols, key='group_col')
        value_map = {}

        if group_col and group_col in df.columns:
            unique_vals = list(df[group_col].dropna().unique())
            selected_vals = st.multiselect("분석할 값을 선택하세요", unique_vals, default=unique_vals[:2], key='group_values')
            if selected_vals:
                col1, col2 = st.columns([2, 6])
                for val in selected_vals:
                    with col1:
                        st.write(f"값: {val}")
                    with col2:
                        label = st.text_input(f"해당 값의 표시 라벨", value=str(val), key=f'label_{val}')
                        value_map[val] = label

                if st.button("논문 Table1 생성", key='table1_generate'):
                    target_df = df.dropna(subset=[group_col])
                    result = analyze_table1_display(target_df, group_col, value_map, threshold=20)
                    st.dataframe(result, use_container_width=True)

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
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

        time_col  = st.selectbox("생존기간 변수(time)", df.columns, key="cox_time_col")
        event_col = st.selectbox("Event 변수(예: 0=생존, 1=사망 등)", df.columns, key="cox_event_col")

        temp_df = df.copy()
        if event_col:
            unique_events = list(df[event_col].dropna().unique())
            st.write(f"이 변수의 실제 값: {unique_events}")
            selected_event    = st.multiselect("이벤트(사건) 값", unique_events, key='selected_event_val')
            selected_censored = st.multiselect("생존/관찰종결(censored) 값", unique_events, key='selected_censored_val')
            st.caption("※ 사건값과 검열값은 서로 겹치면 안 됩니다.")
            temp_df["__event_for_cox"] = ensure_binary_event(temp_df[event_col], set(selected_event), set(selected_censored))
        else:
            temp_df["__event_for_cox"] = np.nan

        candidate_vars = [c for c in df.columns if c not in [time_col, event_col]]
        variables = st.multiselect("분석 후보 변수 선택", candidate_vars, key="cox_variables")

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            p_enter = st.number_input("다변량 포함 기준 p-enter (≤)", min_value=0.001, max_value=1.0, value=0.05, step=0.01)
        with c2:
            max_levels = st.number_input("범주형 판정 최대 고유값", min_value=2, max_value=50, value=10, step=1)
        with c3:
            auto_penal = st.checkbox("penalizer 자동 선택 (CV, C-index)", value=False)  # === NEW
        with c4:
            cv_k = st.number_input("CV folds (K)", min_value=3, max_value=10, value=5, step=1, disabled=not auto_penal)  # === NEW

        penalizer = st.number_input("penalizer (수렴 안정화)", min_value=0.0, max_value=5.0, value=0.1, step=0.1, disabled=auto_penal)

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
            n_total  = temp_df2.shape[0]
            st.info(f"총 관측치: {n_total}, 이벤트 수: {n_events}")
            if n_events < 5:
                st.warning("이벤트 수가 <5로 매우 적습니다. 추정이 불안정하거나 모델이 실패할 수 있습니다.")

            # ---------- 1) Univariate ----------
            uni_sum_dict = {}
            uni_na_vars  = []
            cat_info     = {}

            for var in variables:
                try:
                    dat_raw = temp_df2[[time_col, "__event_for_cox", var]].copy()
                    dat_raw = dat_raw.dropna(subset=[var])
                    if dat_raw.empty:
                        uni_na_vars.append(var); continue

                    if (dat_raw[var].dtype == "object") or (dat_raw[var].nunique(dropna=True) <= max_levels):
                        lvls = ordered_levels(dat_raw[var])
                        if len(lvls) < 2:
                            uni_na_vars.append(var); continue
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
                            if len(lvls) < 2:
                                continue
                            xi = make_dummies(temp_df2[[var]], var, lvls)
                        X_list.append(xi)

                    if not X_list:
                        multi_na_vars = selected_vars
                    else:
                        X_all = pd.concat([dat_base] + X_list, axis=1).dropna()
                        X_all = drop_constant_predictors(X_all, time_col, "__event_for_cox")

                        # === NEW: Auto-CV로 penalizer 선택 ===
                        if auto_penal and X_all["__event_for_cox"].sum() >= int(cv_k):
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
            st.dataframe(result_table, use_container_width=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result_table.to_excel(writer, index=False)
            st.download_button(
                label="Cox 결과 엑셀로 저장",
                data=output.getvalue(),
                file_name="Cox_Regression_Results_Table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # ===== TAB3: Logistic Regression (NEW) =====
    with tab3:
        # --- 지역 임포트: 상단 임포트는 건드리지 않습니다 ---
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import PerfectSeparationError

        st.header("로지스틱 회귀분석 (Risk Factor Analysis)")
        st.info("종속변수의 특정 값을 사건(1)과 기준(0)으로 정의하여 위험인자를 분석합니다. Cox 테이블과 동일한 논문 형식으로 결과를 만듭니다.")

        # ---- Hosmer–Lemeshow 검정 함수 (탭 내부에 정의; 전역 오염 방지) ----
        def calculate_hosmer_lemeshow(y_true, y_prob, g=10):
            """Return (HL_stat, p_value, error_message). g=number of bins."""
            import numpy as np
            from scipy import stats as _stats
            try:
                y_true = np.asarray(y_true, dtype=float)
                y_prob = np.asarray(y_prob, dtype=float)
                mask = np.isfinite(y_true) & np.isfinite(y_prob)
                y_true, y_prob = y_true[mask], y_prob[mask]
                if y_true.size < 20:
                    return None, None, "표본 수가 너무 적습니다(<20)."
                # deciles by predicted probability
                try:
                    bins = pd.qcut(y_prob, q=min(g, max(2, np.unique(y_prob).size)), duplicates='drop')
                except Exception:
                    q = min(g, 5)
                    bins = pd.qcut(y_prob, q=q, duplicates='drop')
                df_hl = pd.DataFrame({'y': y_true, 'p': y_prob, 'bin': bins})
                grp = df_hl.groupby('bin', observed=True)
                O1 = grp['y'].sum()
                N  = grp['y'].count()
                Pbar = grp['p'].mean()
                E1 = N * Pbar
                O0 = N - O1
                E0 = N - E1
                eps = 1e-12
                HL = (((O1 - E1)**2)/(E1 + eps) + ((O0 - E0)**2)/(E0 + eps)).sum()
                df = max(1, len(grp) - 2)
                p  = 1 - _stats.chi2.cdf(HL, df)
                return float(HL), float(p), None
            except Exception as e:
                return None, None, f"HL 계산 오류: {type(e).__name__}: {e}"

        # ---- UI ----
        dep_var = st.selectbox("종속 변수 (Y) 선택", df.columns, key="logistic_dep_var")

        if dep_var:
            unique_outcomes = list(df[dep_var].dropna().unique())
            st.write(f"'{dep_var}' 변수의 고유 값: {unique_outcomes}")
            event_values   = st.multiselect("사건(Event=1)에 해당하는 값 선택", unique_outcomes, key="logistic_event")
            control_values = st.multiselect("기준(Control=0)에 해당하는 값 선택", unique_outcomes, key="logistic_control")
            st.caption("※ 사건 값과 기준 값은 서로 겹치면 안 됩니다.")
        else:
            event_values, control_values = [], []

        indep_vars = st.multiselect(
            "독립 변수 (X) 선택 (위험인자 후보)", [c for c in df.columns if c != dep_var], key="logistic_indep_vars"
        )

        c1_log, c2_log = st.columns(2)
        p_enter_logistic = c1_log.number_input(
            "다변량 포함 기준 p-enter (≤)", min_value=0.001, max_value=1.0, value=0.05, step=0.01, key='logistic_p_enter'
        )
        max_levels_logistic = c2_log.number_input(
            "범주형 판정 최대 고유값", min_value=2, max_value=50, value=10, step=1, key="logistic_max_levels"
        )

        if st.button("로지스틱 회귀분석 실행", key="run_logistic"):
            # ---- 필수 검증 ----
            if not dep_var or not event_values or not control_values or not indep_vars:
                st.error("종속 변수, 사건 값, 기준 값, 독립 변수를 모두 선택해야 합니다.")
                st.stop()
            if set(event_values) & set(control_values):
                st.error("사건 값과 기준 값이 겹칩니다. 다시 선택하세요.")
                st.stop()

            try:
                with st.spinner('분석을 수행 중입니다...'):
                    # ---- 데이터 준비: Y 이진화 ----
                    cols_to_use = [dep_var] + indep_vars
                    df_model = df[cols_to_use].copy()
                    df_model['__y_bin'] = ensure_binary_event(df_model[dep_var], set(event_values), set(control_values))
                    df_model.dropna(subset=['__y_bin'], inplace=True)
                    df_model['__y_bin'] = df_model['__y_bin'].astype(int)

                    # ---- X 전처리 (범주형 더미, 연속형 숫자화) ----
                    X_list, cat_info_logistic = [], {}
                    for var in indep_vars:
                        # 범주형 판정: 고유값 ≤ max_levels OR dtype=object → 범주형
                        if (df_model[var].dtype == 'object') or (df_model[var].nunique(dropna=True) <= max_levels_logistic):
                            levels = ordered_levels(df_model[var])
                            if len(levels) < 2:
                                continue
                            cat_info_logistic[var] = {"levels": levels, "ref": levels[0]}
                            X_list.append(make_dummies(df_model[[var]], var, levels))  # drop_first=True 내부 구현
                        else:
                            cat_info_logistic[var] = {"levels": None, "ref": None}
                            X_list.append(pd.to_numeric(df_model[var], errors='coerce').rename(var))

                    if not X_list:
                        st.error("유효한 독립 변수가 없습니다.")
                        st.stop()

                    X_processed = pd.concat(X_list, axis=1)
                    X_processed = X_processed.loc[:, X_processed.nunique(dropna=True) > 1]  # 상수 열 제거
                    model_data = pd.concat([df_model['__y_bin'], X_processed], axis=1).dropna()
                    y_final = model_data['__y_bin']
                    X_final = model_data.drop(columns=['__y_bin'])
                    X_final = sm.add_constant(X_final, has_constant='add')  # const 추가

                    if X_final.shape[1] <= 1 or len(np.unique(y_final)) < 2:
                        st.error("분석에 사용할 유효한 독립 변수/반응값이 부족합니다.")
                        st.stop()

                    st.info(f"분석에 사용된 총 관측치: {int(len(y_final))}, 사건 수: {int(y_final.sum())}")

                    # ---------- 1) Univariate ----------
                    uni_results = {}           # {var: fitted_result}
                    univariate_pmins = {}      # {var: min p among its terms}
                    uni_fail_reasons = {}      # {var: reason}

                    for var in indep_vars:
                        try:
                            # 해당 변수의 모든 컬럼(연속형 1개 또는 더미들)만 추출
                            var_cols = [c for c in X_final.columns if (c == var) or c.startswith(f"{var}=")]
                            if not var_cols:
                                continue
                            X_uni = X_final[['const'] + var_cols].copy()
                            y_uni = y_final.loc[X_uni.index]
                            # 유효성
                            if X_uni.shape[1] <= 1 or len(np.unique(y_uni)) < 2:
                                uni_fail_reasons[var] = "유효하지 않은 설계행렬/반응값"
                                continue
                            # 적합 (newton → 실패 시 lbfgs 재시도)
                            try:
                                res = sm.Logit(y_uni, X_uni).fit(method='newton', disp=0)
                            except Exception:
                                res = sm.Logit(y_uni, X_uni).fit(method='lbfgs', disp=0)
                            uni_results[var] = res
                            # min p (const 제외)
                            pvals = [res.pvalues[c] for c in res.pvalues.index if c != 'const']
                            if pvals:
                                univariate_pmins[var] = float(np.nanmin(pvals))
                        except PerfectSeparationError:
                            uni_fail_reasons[var] = "데이터 완전 분리(Perfect Separation)"
                        except np.linalg.LinAlgError:
                            uni_fail_reasons[var] = "다중공선성(Singular Matrix)"
                        except Exception as e:
                            uni_fail_reasons[var] = f"오류: {type(e).__name__}: {e}"

                    if uni_fail_reasons:
                        with st.expander("단변량 적합 실패 변수/사유 보기"):
                            for v, r in uni_fail_reasons.items():
                                st.caption(f"- **{v}** → {r}")

                    # ---------- 2) 변수 선택 (p-enter 기준) ----------
                    selected_vars_for_multi = [v for v, p in univariate_pmins.items() if p <= p_enter_logistic]
                    st.write(f"**다변량 분석 포함 변수 (p ≤ {p_enter_logistic:.3f})**: {selected_vars_for_multi if selected_vars_for_multi else '없음'}")

                    # ---------- 3) Multivariate ----------
                    result_multi = None
                    X_multi, y_multi = None, None
                    if selected_vars_for_multi:
                        multi_cols = ['const']
                        for var in selected_vars_for_multi:
                            multi_cols += [c for c in X_final.columns if (c == var) or c.startswith(f"{var}=")]
                        multi_cols = list(dict.fromkeys(multi_cols))  # 중복 제거 & 순서 유지
                        X_multi = X_final[multi_cols].copy()
                        y_multi = y_final.loc[X_multi.index]
                        if X_multi.shape[1] > 1 and len(np.unique(y_multi)) > 1:
                            try:
                                result_multi = sm.Logit(y_multi, X_multi).fit(method='newton', disp=0)
                            except Exception:
                                result_multi = sm.Logit(y_multi, X_multi).fit(method='lbfgs', disp=0)

                    # ---------- 4) 논문용 표 생성 (Cox 형식과 유사) ----------
                    rows = []
                    for var in indep_vars:
                        # 섹션 헤더
                        rows.append({
                            'Factor': var, 'Subgroup': '',
                            'Univariate OR (95% CI)': '', 'Univariate p-Value': '',
                            'Multivariate OR (95% CI)': '', 'Multivariate p-Value': ''
                        })

                        is_cat = (cat_info_logistic.get(var, {}).get('levels') is not None)
                        if is_cat:
                            levels = cat_info_logistic[var]['levels']
                            ref    = cat_info_logistic[var]['ref']
                            # Reference 라인
                            rows.append({
                                'Factor': '', 'Subgroup': f'{ref} (Reference)',
                                'Univariate OR (95% CI)': 'Ref.', 'Univariate p-Value': '',
                                'Multivariate OR (95% CI)': 'Ref.', 'Multivariate p-Value': ''
                            })
                            # 각 더미 레벨
                            for lv in levels[1:]:
                                term = f"{var}={lv}"
                                # Univariate
                                if (var in uni_results) and (term in uni_results[var].params.index):
                                    r = uni_results[var]
                                    b = float(r.params[term]); p = float(r.pvalues[term])
                                    ci_low, ci_high = [float(x) for x in r.conf_int().loc[term].tolist()]
                                    or_txt = f"{np.exp(b):.3f} ({np.exp(ci_low):.3f}-{np.exp(ci_high):.3f})"
                                    p_txt  = format_p(p)
                                else:
                                    or_txt, p_txt = 'NA', 'NA'
                                # Multivariate
                                if (result_multi is not None) and (term in result_multi.params.index):
                                    r = result_multi
                                    b = float(r.params[term]); p = float(r.pvalues[term])
                                    ci_low, ci_high = [float(x) for x in r.conf_int().loc[term].tolist()]
                                    or_multi = f"{np.exp(b):.3f} ({np.exp(ci_low):.3f}-{np.exp(ci_high):.3f})"
                                    p_multi  = format_p(p)
                                else:
                                    or_multi, p_multi = 'NA', 'NA'
                                rows.append({
                                    'Factor': '', 'Subgroup': str(lv),
                                    'Univariate OR (95% CI)': or_txt, 'Univariate p-Value': p_txt,
                                    'Multivariate OR (95% CI)': or_multi, 'Multivariate p-Value': p_multi
                                })
                        else:
                            # 연속형 변수 1열
                            if (var in uni_results) and (var in uni_results[var].params.index):
                                r = uni_results[var]
                                b = float(r.params[var]); p = float(r.pvalues[var])
                                ci_low, ci_high = [float(x) for x in r.conf_int().loc[var].tolist()]
                                or_txt = f"{np.exp(b):.3f} ({np.exp(ci_low):.3f}-{np.exp(ci_high):.3f})"
                                p_txt  = format_p(p)
                            else:
                                or_txt, p_txt = 'NA', 'NA'
                            if (result_multi is not None) and (var in result_multi.params.index):
                                r = result_multi
                                b = float(r.params[var]); p = float(r.pvalues[var])
                                ci_low, ci_high = [float(x) for x in r.conf_int().loc[var].tolist()]
                                or_multi = f"{np.exp(b):.3f} ({np.exp(ci_low):.3f}-{np.exp(ci_high):.3f})"
                                p_multi  = format_p(p)
                            else:
                                or_multi, p_multi = 'NA', 'NA'
                            rows.append({
                                'Factor': '', 'Subgroup': '',
                                'Univariate OR (95% CI)': or_txt, 'Univariate p-Value': p_txt,
                                'Multivariate OR (95% CI)': or_multi, 'Multivariate p-Value': p_multi
                            })

                    publication_df = pd.DataFrame(rows)
                    st.write("**논문 제출용 테이블 (Univariate/Multivariate 병렬, Reference, Factor/수준구조)**")
                    st.dataframe(publication_df, use_container_width=True)

                    # ---- HL 적합도 ----
                    if result_multi is not None:
                        st.write("---")
                        st.subheader("모델 적합도: Hosmer–Lemeshow Test")
                        y_pred_prob = result_multi.predict(X_multi)
                        hl_stat, hl_p, hl_err = calculate_hosmer_lemeshow(y_multi, y_pred_prob)
                        if hl_err:
                            st.warning(f"호스머–렘쇼 검정을 수행할 수 없습니다: {hl_err}")
                        else:
                            c1, c2 = st.columns(2)
                            c1.metric("Chi-squared statistic", f"{hl_stat:.3f}")
                            c2.metric("p-value", f"{hl_p:.3f}")
                            st.caption("※ p-value가 0.05보다 크면 모델이 데이터에 적절히 적합한다고 볼 수 있습니다.")

                    # ---- 엑셀 다운로드 ----
                    output_logistic = io.BytesIO()
                    with pd.ExcelWriter(output_logistic, engine='openpyxl') as writer:
                        publication_df.to_excel(writer, index=False, sheet_name='Logistic Regression Results')
                    st.download_button(
                        label="로지스틱 결과 엑셀로 저장",
                        data=output_logistic.getvalue(),
                        file_name="Logistic_Regression_Publication_Table.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key='download_logistic_publication'
                    )

            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {type(e).__name__}: {e}")

# ================== 끝 ==============================
