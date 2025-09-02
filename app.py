# -*- coding: utf-8 -*-
# ================== 자동 Table & Cox & Logistic 분석기 ==================
# 필요 패키지: pandas, numpy, scipy, lifelines, statsmodels, openpyxl, xlrd, streamlit

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import io
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
import statsmodels.api as sm # For Logistic Regression
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import matplotlib.pyplot as plt

# ----- 페이지 설정 -----
st.set_page_config(page_title="자동 논문 Table", layout="wide")

# ----- 간단 비밀번호 보호 (기본: CRCR 또는 st.secrets['APP_PASSWORD']) -----
def _check_password():
    def _password_entered():
        # Streamlit secrets에 저장된 비밀번호 또는 기본값 "CRCR" 사용
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

# 비밀번호 체크 실행
_check_password()

st.title("자동 Table 생성기")

# -------------------- [공통 유틸리티 함수] --------------------
def format_p(p):
    """p-value를 논문 형식에 맞게 변환합니다."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p >= 0.999:
        return "p > 0.99"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def is_continuous(series, threshold=20):
    """시리즈가 연속형 변수인지 판별합니다."""
    try:
        # 숫자형(정수, 실수)이고 고유값 개수가 threshold를 초과하면 연속형으로 판단
        return (series.dtype.kind in "fi") and (series.nunique(dropna=True) > threshold)
    except Exception:
        return False

def ordered_levels(series):
    """범주형 변수의 레벨을 정렬합니다. 숫자형이면 숫자 순, 아니면 문자 순으로 정렬합니다."""
    # 고유값을 문자열로 변환하여 일관성 유지 (예: 1과 '1'을 동일하게 처리)
    unique_strings = pd.Series(series.dropna().unique()).astype(str).unique().tolist()
    
    try:
        # 숫자(float) 기준으로 정렬 시도
        unique_strings.sort(key=float)
    except ValueError:
        # 하나라도 float 변환 실패 시, 문자열 기준으로 정렬
        unique_strings.sort()
    
    return unique_strings

def make_dummies(df_in, var, levels):
    """범주형 변수를 더미 변수로 변환합니다. 첫 번째 레벨을 기준으로 합니다."""
    cat = pd.Categorical(df_in[var].astype(str),
                         categories=[str(x) for x in levels],
                         ordered=True)
    dmy = pd.get_dummies(cat, prefix=var, prefix_sep="=", drop_first=True, dtype=float)
    dmy.index = df_in.index
    return dmy

def dummy_colname(var, level):
    """더미 변수의 컬럼명을 생성합니다."""
    return f"{var}={str(level)}"

def drop_constant_cols(X):
    """상수 컬럼을 제거하되, 회귀분석에 필요한 'const' 컬럼은 유지합니다."""
    cols_to_keep = []
    for col in X.columns:
        # 'const' 컬럼이거나, 고유값이 1개 초과인 경우에만 유지
        if col == 'const' or X[col].nunique(dropna=True) > 1:
            cols_to_keep.append(col)
    return X[cols_to_keep]

def drop_constant_predictors(X, time_col, event_col):
    """Cox 분석용 데이터에서 상수인 예측 변수를 제거합니다."""
    pred_cols = [c for c in X.columns if c not in [time_col, event_col]]
    keep = [c for c in pred_cols if X[c].nunique(dropna=True) > 1]
    return X[[time_col, event_col] + keep]

def clean_time(s):
    """생존 분석의 시간 변수를 정리합니다. (숫자형 변환, inf/nan 처리)"""
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def ensure_binary_event(col, events, censored):
    """이벤트 변수를 0(censored)과 1(event)로 변환합니다."""
    def _map(x):
        if x in events: return 1
        if x in censored: return 0
        return np.nan
    return col.apply(_map).astype(float)

def calculate_hosmer_lemeshow(y_true, y_pred_prob, n_groups=10):
    """로지스틱 회귀분석의 Hosmer-Lemeshow 적합도 검정을 계산합니다."""
    data = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
    
    try:
        data['group'] = pd.qcut(data['y_pred_prob'], q=n_groups, duplicates='drop')
    except ValueError:
        # 중복 예측 확률값으로 인해 그룹 나누기 실패 시, 그룹 수를 줄여서 재시도
        n_groups = data['y_pred_prob'].nunique()
        if n_groups < 2: return np.nan, np.nan, "Not enough unique probabilities for the test."
        data['group'] = pd.qcut(data['y_pred_prob'], q=n_groups, duplicates='drop')
    
    summary = data.groupby('group', observed=False).agg(
        total_count=('y_true', 'count'),
        observed_events=('y_true', 'sum'),
        expected_events=('y_pred_prob', 'sum')
    )
    
    summary['observed_non_events'] = summary['total_count'] - summary['observed_events']
    summary['expected_non_events'] = summary['total_count'] - summary['expected_events']

    if (summary['expected_events'] < 1e-6).any() or (summary['expected_non_events'] < 1e-6).any():
        return np.nan, np.nan, "Test failed due to zero or near-zero expected frequencies in some groups."

    hl_stat = (
        ((summary['observed_events'] - summary['expected_events'])**2 / summary['expected_events']) +
        ((summary['observed_non_events'] - summary['expected_non_events'])**2 / summary['expected_non_events'])
    ).sum()

    df_hl = len(summary) - 2
    if df_hl <= 0:
        return np.nan, np.nan, "Not enough groups to calculate p-value (degrees of freedom <= 0)."
        
    p_value = stats.chi2.sf(hl_stat, df_hl)
    
    return hl_stat, p_value, None

def select_penalizer_by_cv(X_all, time_col, event_col,
                           grid=(0.0, 0.01, 0.05, 0.1, 0.2, 0.5),
                           k=5, seed=42):
    """교차 검증(Cross-Validation)을 통해 최적의 penalizer 값을 찾습니다."""
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
            train = drop_constant_predictors(train, time_col, event_col)
            test  = test[train.columns]
            if train[event_col].sum() < 2 or test[event_col].sum() < 1: continue
            if train.shape[1] <= 2 or train.shape[0] < 5: continue
            try:
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(train, duration_col=time_col, event_col=event_col)
                s = float(cph.score(test, scoring_method="concordance_index"))
                if np.isfinite(s): cv_scores.append(s)
            except Exception: continue
        if cv_scores: scores[pen] = float(np.mean(cv_scores))
    if not scores: return None, {}
    best_pen = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return best_pen, scores

# -------------------- [파일 업로드 UI] --------------------
uploaded_file = st.file_uploader("엑셀/CSV 업로드", type=['xls', 'xlsx', 'csv'])
df = None
sheetname = None
if uploaded_file:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            engine = "openpyxl" if name.endswith(".xlsx") else "xlrd"
            xls = pd.ExcelFile(uploaded_file, engine=engine)
            sheetname = xls.sheet_names[0] if len(xls.sheet_names) == 1 else st.selectbox("시트 선택", xls.sheet_names, index=0)
            df = pd.read_excel(xls, sheet_name=sheetname, engine=engine)
        
        # 컬럼명 공백 및 개행 문자 제거
        df.columns = pd.Index([str(c).strip().replace("\\n", " ") for c in df.columns])
        st.success(f"시트명: {sheetname if sheetname else ''}, 데이터 shape: {df.shape}")
        st.dataframe(df.head())
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

# -------------------- [Table1: 분석 함수] --------------------
def analyze_table1_display(df, group_col, value_map, threshold=20):
    result_rows = []
    group_values = list(value_map.keys())
    group_names = list(value_map.values())
    group_n = {g: (df[group_col] == g).sum() for g in group_values}
    
    for var in df.columns:
        if var == group_col: continue
        valid = df[df[group_col].isin(group_values)]
        if valid[var].dropna().empty: continue
        
        if is_continuous(valid[var], threshold=threshold):
            # 연속형 변수 처리
            row = {'Characteristic': var}
            for g, g_name in zip(group_values, group_names):
                sub = valid[valid[group_col] == g][var].dropna()
                if sub.shape[0] > 0:
                    med, q1, q3 = sub.median(), sub.quantile(0.25), sub.quantile(0.75)
                    mean, std = sub.mean(), sub.std()
                    row[f"{g_name} (n={group_n[g]})"] = f"{med:.1f} [{q1:.1f}-{q3:.1f}]; {mean:.1f}±{std:.1f}"
                else:
                    row[f"{g_name} (n={group_n[g]})"] = "NA"
            
            p = np.nan; test_str = ""
            try:
                arr = [valid[valid[group_col] == g][var].dropna() for g in group_values]
                # 정규성 검정 후 t-test/ANOVA 또는 비모수 검정 선택
                if all(stats.shapiro(s)[1] > 0.05 for s in arr if len(s) >= 3):
                    if len(arr) == 2: _, p = stats.ttest_ind(arr[0], arr[1], nan_policy='omit'); test_str = "t-test"
                    elif len(arr) > 2: _, p = stats.f_oneway(*arr); test_str = "ANOVA"
                else:
                    if len(arr) == 2: _, p = stats.mannwhitneyu(arr[0], arr[1], alternative='two-sided'); test_str = "Mann-Whitney U"
                    elif len(arr) > 2: _, p = stats.kruskal(*arr); test_str = "Kruskal-Wallis"
            except Exception: p = np.nan; test_str = "NA"
            row['Test'] = test_str; row['p value'] = format_p(p); row['sub_p'] = ""
            result_rows.append(row)
        else:
            # 범주형 변수 처리
            p = np.nan; test_str = ""
            try:
                ct = pd.crosstab(valid[group_col], valid[var])
                if ct.shape == (2, 2): _, p = stats.fisher_exact(ct); test_str = "Fisher's Exact"
                else: _, p, _, _ = stats.chi2_contingency(ct); test_str = "Chi-square"
            except Exception: p = np.nan; test_str = "NA"
            
            result_rows.append({'Characteristic': var, **{f"{g_name} (n={group_n[g]})": "" for g_name in group_names}, 'Test': test_str, 'p value': format_p(p), 'sub_p': ""})
            
            for val in ordered_levels(valid[var]):
                row = {'Characteristic': f"  {val}"}
                for g, g_name in zip(group_values, group_names):
                    cnt = valid[(valid[group_col] == g) & (valid[var].astype(str) == str(val))].shape[0]
                    percent = (cnt/group_n[g]*100) if group_n[g] > 0 else 0
                    row[f"{g_name} (n={group_n[g]})"] = f"{cnt}({percent:.0f}%)"
                
                row['Test'] = ""
                row['p value'] = ""
                p_sub = np.nan
                try:
                    # 하위 그룹 p-value 계산
                    table = np.array([[(valid[(valid[group_col] == g) & (valid[var].astype(str) == str(val))]).shape[0], (valid[(valid[group_col] == g) & (valid[var].astype(str) != str(val))]).shape[0]] for g in group_values])
                    if table.shape == (2, 2): _, p_sub = stats.fisher_exact(table)
                    else: _, p_sub, _, _ = stats.chi2_contingency(table)
                except: pass
                row['sub_p'] = format_p(p_sub)
                result_rows.append(row)
                
    return pd.DataFrame(result_rows)


# -------------------- [UI: 탭 구성] --------------------
if 'df' in st.session_state:
    df = st.session_state['df']
else:
    df = None

if df is not None:
    tab_titles = ["📊 Table1 자동화", "🟦 Cox 회귀분석", "🟧 로지스틱 회귀분석"]
    tab1, tab2, tab3 = st.tabs(tab_titles)

    # ==================== TAB1: Table 1 ====================
    with tab1:
        st.header("Table 1 자동 생성")
        group_col = st.selectbox("분석할 그룹 변수명 선택", options=list(df.columns), key='group_col')
        if group_col:
            unique_vals = list(df[group_col].dropna().unique())
            selected_vals = st.multiselect("분석할 그룹 값 선택", unique_vals, default=unique_vals[:2] if len(unique_vals) >= 2 else unique_vals, key='group_values')
            if selected_vals:
                value_map = {val: st.text_input(f"'{val}'의 표시 라벨", value=str(val), key=f'label_{val}') for val in selected_vals}
                if st.button("논문 Table1 생성", key='table1_generate'):
                    with st.spinner('Table 1을 생성 중입니다...'):
                        result = analyze_table1_display(df, group_col, value_map, threshold=20)
                        st.dataframe(result)
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            result.to_excel(writer, index=False)
                        st.download_button("Table1 엑셀로 저장", output.getvalue(), "Table1_Results.xlsx")

    # ==================== TAB2: Cox Regression ====================
    with tab2:
        st.header("Cox 비례위험 회귀분석 (Univariate & Multivariate)")
        time_col = st.selectbox("생존기간 변수(time)", df.columns, key="cox_time_col")
        event_col = st.selectbox("Event 변수(예: 0=생존, 1=사망 등)", df.columns, key="cox_event_col")
        
        if event_col:
            unique_events = list(df[event_col].dropna().unique())
            selected_event = st.multiselect("이벤트(사건) 값", unique_events, key='selected_event_val')
            selected_censored = st.multiselect("생존/관찰종결(censored) 값", unique_events, key='selected_censored_val')
            
        variables = st.multiselect("분석 후보 변수 선택", [c for c in df.columns if c not in [time_col, event_col]], key="cox_variables")
        
        c1, c2, c3, c4 = st.columns(4)
        p_enter = c1.number_input("다변량 포함 기준 p-enter (≤)", 0.001, 1.0, 0.05, 0.01)
        max_levels = c2.number_input("범주형 판정 최대 고유값", 2, 50, 10, 1, key="cox_max_levels")
        auto_penal = c3.checkbox("penalizer 자동 선택 (CV)", value=False)
        cv_k = c4.number_input("CV folds (K)", 3, 10, 5, 1, disabled=not auto_penal)
        penalizer = st.number_input("penalizer (수렴 안정화)", 0.0, 5.0, 0.1, 0.01, disabled=auto_penal)

        if st.button("Cox 회귀분석 실행"):
            with st.spinner('Cox 회귀분석을 수행 중입니다...'):
                if not selected_event or not selected_censored:
                    st.error("사건값과 검열값을 각각 최소 1개 이상 선택하세요.")
                    st.stop()
                if set(selected_event) & set(selected_censored):
                    st.error("사건값과 검열값이 겹칩니다. 다시 선택하세요.")
                    st.stop()

                # 데이터 준비
                df_cox = df.copy()
                df_cox["__event_for_cox"] = ensure_binary_event(df_cox[event_col], set(selected_event), set(selected_censored))
                df_cox[time_col] = clean_time(df_cox[time_col])
                df_cox = df_cox.dropna(subset=[time_col, "__event_for_cox"])
                df_cox = df_cox[df_cox[time_col] > 0]
                
                n_events = int(df_cox["__event_for_cox"].sum())
                n_total = df_cox.shape[0]
                st.info(f"총 관측치: {n_total}, 이벤트 수: {n_events}")
                if n_events < 5:
                    st.warning("이벤트 수가 5개 미만으로 매우 적어 분석이 불안정할 수 있습니다.")

                # 1. 단변량 분석 (Univariate)
                uni_results = {}
                failed_uni_vars = {}
                cat_info = {}
                for var in variables:
                    try:
                        cols_to_use = [time_col, "__event_for_cox", var]
                        df_uni = df_cox[cols_to_use].dropna(subset=[var]).copy()
                        if df_uni.shape[0] < 3: continue
                        
                        if is_continuous(df_uni[var], threshold=max_levels):
                            cat_info[var] = {"levels": None, "ref": None}
                            df_uni[var] = pd.to_numeric(df_uni[var], errors='coerce')
                        else:
                            levels = ordered_levels(df_uni[var])
                            cat_info[var] = {"levels": levels, "ref": levels[0]}
                            dummies = make_dummies(df_uni, var, levels)
                            df_uni = pd.concat([df_uni[[time_col, "__event_for_cox"]], dummies], axis=1)
                        
                        df_uni = df_uni.dropna()
                        df_uni = drop_constant_predictors(df_uni, time_col, "__event_for_cox")

                        if df_uni.shape[1] > 2 and df_uni["__event_for_cox"].sum() > 0:
                            cph = CoxPHFitter(penalizer=penalizer)
                            cph.fit(df_uni, duration_col=time_col, event_col="__event_for_cox")
                            uni_results[var] = cph.summary
                    except Exception as e:
                        failed_uni_vars[var] = str(e)
                
                # 2. 다변량 분석을 위한 변수 선택
                univariate_pvals = {var: res['p'].min() for var, res in uni_results.items()}
                selected_vars = [v for v, p in univariate_pvals.items() if p <= p_enter]
                st.write(f"**다변량 분석 포함 변수 (p ≤ {p_enter})**: {selected_vars if selected_vars else '없음'}")

                # 3. 다변량 분석 (Multivariate)
                multi_summary = None
                chosen_penalizer = penalizer
                if selected_vars:
                    try:
                        cols_for_multi = [time_col, "__event_for_cox"] + selected_vars
                        df_multi_raw = df_cox[cols_for_multi].copy()
                        
                        X_list = []
                        for var in selected_vars:
                            if cat_info.get(var, {}).get("levels"):
                                X_list.append(make_dummies(df_multi_raw[[var]], var, cat_info[var]['levels']))
                            else:
                                X_list.append(pd.to_numeric(df_multi_raw[var], errors='coerce').rename(var))
                        
                        X_processed = pd.concat(X_list, axis=1)
                        df_multi = pd.concat([df_multi_raw[[time_col, "__event_for_cox"]], X_processed], axis=1).dropna()
                        df_multi = drop_constant_predictors(df_multi, time_col, "__event_for_cox")

                        if auto_penal:
                            best_pen, scores = select_penalizer_by_cv(df_multi, time_col, "__event_for_cox", k=cv_k)
                            if best_pen is not None:
                                chosen_penalizer = best_pen
                                st.success(f"CV를 통해 최적 Penalizer = {chosen_penalizer} 선택 (C-index 기준)")
                                st.caption(f"Grid 성능: { {k: round(v,4) for k,v in scores.items()} }")
                            else:
                                st.warning("CV로 Penalizer를 결정하지 못해 기본값을 사용합니다.")
                        
                        if df_multi.shape[1] > 2 and df_multi["__event_for_cox"].sum() > 0:
                            cph_multi = CoxPHFitter(penalizer=chosen_penalizer)
                            cph_multi.fit(df_multi, duration_col=time_col, event_col="__event_for_cox")
                            multi_summary = cph_multi.summary
                    except Exception as e:
                        st.error(f"다변량 분석 중 오류가 발생했습니다: {e}")

                # 4. 결과 테이블 생성
                output_rows = []
                for var in variables:
                    is_cat = cat_info.get(var, {}).get('levels') is not None
                    
                    if is_cat:
                        output_rows.append({'Factor': var, 'Subgroup': '', 'Univariate HR (95% CI)': '', 'p (Uni)': '', 'Multivariate HR (95% CI)': '', 'p (Multi)': ''})
                        levels = cat_info[var]['levels']
                        output_rows.append({'Factor': '', 'Subgroup': f"{levels[0]} (Reference)", 'Univariate HR (95% CI)': '1.0', 'p (Uni)': '', 'Multivariate HR (95% CI)': '1.0', 'p (Multi)': ''})
                        
                        for level in levels[1:]:
                            dummy_name = dummy_colname(var, level)
                            row = {'Factor': '', 'Subgroup': str(level)}
                            # Uni
                            if var in uni_results and dummy_name in uni_results[var].index:
                                r = uni_results[var].loc[dummy_name]
                                row['Univariate HR (95% CI)'] = f"{r['exp(coef)']:.3f} ({r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f})"
                                row['p (Uni)'] = format_p(r['p'])
                            # Multi
                            if multi_summary is not None and dummy_name in multi_summary.index:
                                r = multi_summary.loc[dummy_name]
                                row['Multivariate HR (95% CI)'] = f"{r['exp(coef)']:.3f} ({r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f})"
                                row['p (Multi)'] = format_p(r['p'])
                            output_rows.append(row)
                    else: # Continuous
                        row = {'Factor': var, 'Subgroup': ''}
                        # Uni
                        if var in uni_results and var in uni_results[var].index:
                            r = uni_results[var].loc[var]
                            row['Univariate HR (95% CI)'] = f"{r['exp(coef)']:.3f} ({r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f})"
                            row['p (Uni)'] = format_p(r['p'])
                        # Multi
                        if multi_summary is not None and var in multi_summary.index:
                            r = multi_summary.loc[var]
                            row['Multivariate HR (95% CI)'] = f"{r['exp(coef)']:.3f} ({r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f})"
                            row['p (Multi)'] = format_p(r['p'])
                        output_rows.append(row)
                
                publication_df = pd.DataFrame(output_rows).fillna('')
                st.dataframe(publication_df)
                
                output_cox = io.BytesIO()
                with pd.ExcelWriter(output_cox, engine='openpyxl') as writer:
                    publication_df.to_excel(writer, index=False)
                st.download_button("Cox 분석 결과 엑셀 저장", output_cox.getvalue(), "Cox_Regression_Results.xlsx")


    # ==================== TAB3: Logistic Regression ====================
    with tab3:
        st.header("로지스틱 회귀분석 (Risk Factor Analysis)")
        st.info("종속변수의 특정 값을 사건(1)과 기준(0)으로 정의하여 위험인자를 분석합니다.")

        dep_var = st.selectbox("종속 변수 (Y) 선택", df.columns, key="logistic_dep_var")
        
        if dep_var:
            unique_outcomes = list(df[dep_var].dropna().unique())
            event_values = st.multiselect("사건(Event=1)에 해당하는 값 선택", unique_outcomes, key="logistic_event")
            control_values = st.multiselect("기준(Control=0)에 해당하는 값 선택", unique_outcomes, key="logistic_control")
            st.caption("※ 사건 값과 기준 값은 서로 겹치면 안 됩니다.")

        indep_vars = st.multiselect("독립 변수 (X) 선택 (위험인자 후보)", [c for c in df.columns if c != dep_var], key="logistic_indep_vars")
        
        c1_log, c2_log = st.columns(2)
        p_enter_logistic = c1_log.number_input("다변량 포함 기준 p-enter (≤)", 0.001, 1.0, 0.05, 0.01, key='logistic_p_enter')
        max_levels_logistic = c2_log.number_input("범주형 판정 최대 고유값", 2, 50, 10, 1, key="logistic_max_levels")

        if st.button("로지스틱 회귀분석 실행", key="run_logistic"):
            if not dep_var or not event_values or not control_values or not indep_vars:
                st.error("종속 변수, 사건 값, 기준 값, 독립 변수를 모두 선택해야 합니다.")
                st.stop()
            if set(event_values) & set(control_values):
                st.error("사건 값과 기준 값이 겹칩니다. 다시 선택하세요.")
                st.stop()

            try:
                with st.spinner('분석을 수행 중입니다...'):
                    # 데이터 준비
                    cols_to_use = [dep_var] + indep_vars
                    df_model = df[cols_to_use].copy()
                    df_model['__y_binary'] = ensure_binary_event(df_model[dep_var], set(event_values), set(control_values))
                    df_model.dropna(subset=['__y_binary'], inplace=True)
                    df_model['__y_binary'] = df_model['__y_binary'].astype(int)
                    y = df_model['__y_binary']

                    # 독립변수 처리 (더미 변수화)
                    X_list, cat_info_logistic = [], {}
                    for var in indep_vars:
                        if not is_continuous(df_model[var], threshold=max_levels_logistic):
                            levels = ordered_levels(df_model[var])
                            cat_info_logistic[var] = {"levels": levels, "ref": levels[0]}
                            X_list.append(make_dummies(df_model[[var]], var, levels))
                        else:
                            cat_info_logistic[var] = {"levels": None, "ref": None}
                            X_list.append(pd.to_numeric(df_model[var], errors='coerce').rename(var))
                    
                    if not X_list: st.error("유효한 독립 변수가 없습니다."); st.stop()

                    X_processed = pd.concat(X_list, axis=1)
                    model_data = pd.concat([y, X_processed], axis=1).dropna()
                    y_final = model_data[y.name]
                    X_final_no_const = model_data.drop(columns=[y.name])
                    
                    # 상수항 추가 후 상수 예측변수 제거
                    X_final_with_const = sm.add_constant(X_final_no_const, has_constant='add')
                    X_final = drop_constant_cols(X_final_with_const)

                    if X_final.shape[1] <= 1 or 'const' not in X_final.columns:
                        st.error("분석에 사용할 유효한 독립 변수가 부족합니다."); st.stop()
                    st.info(f"분석에 사용된 총 관측치: {len(y_final)}, 사건 수: {y_final.sum()}")

                    # 1. 단변량 분석
                    uni_results, p_values_uni, failed_vars = {}, {}, {}
                    for var in indep_vars:
                        var_cols = [c for c in X_final.columns if c == var or c.startswith(f"{var}=")]
                        if not var_cols: continue
                        
                        X_uni = X_final[['const'] + var_cols]
                        try:
                            res = sm.Logit(y_final, X_uni).fit(method='newton', disp=0) 
                            uni_results[var] = res
                            p_values_uni[var] = res.pvalues.drop('const').min()
                        except Exception as e: 
                            failed_vars[var] = str(e)

                    # 2. 다변량 분석
                    selected_vars_multi = [v for v, p in p_values_uni.items() if p <= p_enter_logistic]
                    st.write(f"**다변량 분석 포함 변수 (p ≤ {p_enter_logistic})**: {selected_vars_multi if selected_vars_multi else '없음'}")
                    
                    result_multi = None
                    if selected_vars_multi:
                        multi_cols = ['const'] + [c for var in selected_vars_multi for c in X_final.columns if c.startswith(f"{var}=") or c == var]
                        X_multi = X_final[multi_cols]
                        try:
                            result_multi = sm.Logit(y_final, X_multi).fit(method='newton', disp=0)
                        except Exception as e:
                            st.error(f"다변량 분석 중 오류 발생: {e}")

                    # 3. 결과 테이블 생성
                    output_rows = []
                    for var in indep_vars:
                        is_cat = cat_info_logistic[var]['levels'] is not None
                        if is_cat:
                            output_rows.append({'Factor': var, 'Subgroup': '', 'Univariate OR (95% CI)': '', 'p-value (Uni)': '', 'Multivariate OR (95% CI)': '', 'p-value (Multi)': ''})
                            levels = cat_info_logistic[var]['levels']
                            output_rows.append({'Factor': '', 'Subgroup': f"{levels[0]} (Reference)", 'Univariate OR (95% CI)': '1.0', 'p-value (Uni)': '', 'Multivariate OR (95% CI)': '1.0' if var in selected_vars_multi else '', 'p-value (Multi)': ''})
                            for level in levels[1:]:
                                d_name = dummy_colname(var, level)
                                row = {'Factor': '', 'Subgroup': str(level)}
                                if var in uni_results and d_name in uni_results[var].params:
                                    res, p, conf = uni_results[var].params[d_name], uni_results[var].pvalues[d_name], uni_results[var].conf_int().loc[d_name]
                                    row['Univariate OR (95% CI)'] = f"{np.exp(res):.3f} ({np.exp(conf[0]):.3f}-{np.exp(conf[1]):.3f})"
                                    row['p-value (Uni)'] = format_p(p)
                                if result_multi and d_name in result_multi.params:
                                    res, p, conf = result_multi.params[d_name], result_multi.pvalues[d_name], result_multi.conf_int().loc[d_name]
                                    row['Multivariate OR (95% CI)'] = f"{np.exp(res):.3f} ({np.exp(conf[0]):.3f}-{np.exp(conf[1]):.3f})"
                                    row['p-value (Multi)'] = format_p(p)
                                output_rows.append(row)
                        else: # Continuous
                            row = {'Factor': var, 'Subgroup': ''}
                            if var in uni_results and var in uni_results[var].params:
                                res, p, conf = uni_results[var].params[var], uni_results[var].pvalues[var], uni_results[var].conf_int().loc[var]
                                row['Univariate OR (95% CI)'] = f"{np.exp(res):.3f} ({np.exp(conf[0]):.3f}-{np.exp(conf[1]):.3f})"
                                row['p-value (Uni)'] = format_p(p)
                            if result_multi and var in result_multi.params:
                                res, p, conf = result_multi.params[var], result_multi.pvalues[var], result_multi.conf_int().loc[var]
                                row['Multivariate OR (95% CI)'] = f"{np.exp(res):.3f} ({np.exp(conf[0]):.3f}-{np.exp(conf[1]):.3f})"
                                row['p-value (Multi)'] = format_p(p)
                            output_rows.append(row)

                    publication_df = pd.DataFrame(output_rows).fillna('')
                    st.dataframe(publication_df)

                    if result_multi:
                        st.write("---")
                        st.write("### 모델 적합도 검정 (Hosmer-Lemeshow Test)")
                        y_pred_prob = result_multi.predict(X_multi)
                        hl_stat, p_value_hl, hl_error = calculate_hosmer_lemeshow(y_final, y_pred_prob)
                        if hl_error:
                            st.warning(f"호스머-렘쇼 검정을 수행할 수 없습니다: {hl_error}")
                        else:
                            col1, col2 = st.columns(2)
                            col1.metric("Chi-squared statistic", f"{hl_stat:.3f}")
                            col2.metric("p-value", f"{p_value_hl:.3f}")
                            st.caption("※ p-value가 0.05보다 크면 모델이 데이터에 잘 적합한다고 해석할 수 있습니다.")

                    output_logistic = io.BytesIO()
                    with pd.ExcelWriter(output_logistic, engine='openpyxl') as writer:
                        publication_df.to_excel(writer, index=False)
                    st.download_button("로지스틱 분석 결과 엑셀 저장", output_logistic.getvalue(), "Logistic_Regression_Results.xlsx")

            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")
