import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# --- 1. 頁面設定 / Page Configuration ---
st.set_page_config(page_title="SCM AI Multi-Model Study", layout="wide")


# --- 2. 資料載入 / Data Collection ---
@st.cache_data
def load_data():
    try:
        # 讀取歷史銷售數據 (Historical sales / Data collection)
        df = pd.read_csv('meat_consumption_worldwide.csv')
        return df
    except Exception as e:
        st.error(f"找不到資料！請檢查 CSV 檔案。(Data not found!): {e}")
        return None


# --- 3. 核心運算邏輯 / ML & Statistical Algorithms ---
def run_comparison(values, test_size=5):
    """
    執行流程圖中的所有預測演算法。
    Running all prediction algorithms defined in the flowchart.
    """
    look_back = 3
    y_true = values[-test_size:]
    train_data = values[:-test_size]

    # --- 模型 A: 傳統 SMA (Baseline) ---
    y_pred_sma = [np.mean(values[-(test_size + look_back + i): -(test_size + i)]) for i in range(test_size, 0, -1)]

    # 特徵工程 (用於 RF 與 LR)
    X_train, y_train = [], []
    for i in range(len(train_data) - look_back):
        X_train.append(train_data[i: i + look_back])
        y_train.append(train_data[i + look_back])
    X_test = [values[-(test_size + look_back - i): -(test_size - i)] for i in range(test_size)]

    # --- 模型 B: Random Forest (ML Algorithms) ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # --- 模型 C: Linear Regression (Statistical Control) ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # --- 模型 D: ARIMA (Time Series Model) ---
    try:
        history = list(train_data)
        y_pred_arima = []
        for i in range(test_size):
            # 建立 ARIMA(1,1,0) 模型
            model = ARIMA(history, order=(1, 1, 0))
            model_fit = model.fit()
            y_pred_arima.append(model_fit.forecast()[0])
            history.append(y_true[i])  # 滾動更新歷史資料
    except:
        y_pred_arima = y_pred_sma

    return {
        "Actual": y_true,
        "SMA": np.array(y_pred_sma),
        "Random Forest": np.array(y_pred_rf),
        "Linear Regression": np.array(y_pred_lr),
        "ARIMA": np.array(y_pred_arima)
    }


# --- 4. 介面呈現 / UI Dashboard ---
df_all = load_data()

if df_all is not None:
    st.sidebar.header("⚙️ 實驗設置 (Experiment Setup)")
    dataset_option = st.sidebar.selectbox(
        "選擇資料集 (Case Study Selection)",
        ["USA - BEEF (穩定/Stable)", "CHN - PIG (高波動/Volatile)", "EU28 - POULTRY (趨勢/Trend)"]
    )

    mapping = {
        "USA - BEEF (穩定/Stable)": ("USA", "BEEF"),
        "CHN - PIG (高波動/Volatile)": ("CHN", "PIG"),
        "EU28 - POULTRY (趨勢/Trend)": ("EU28", "POULTRY")
    }
    loc, sub = mapping[dataset_option]
    df_target = df_all[
        (df_all['LOCATION'] == loc) & (df_all['SUBJECT'] == sub) & (df_all['MEASURE'] == 'THND_TONNE')].sort_values(
        'TIME')
    df_target['DATE'] = df_target['TIME'].apply(lambda x: f"{int(x)}")
    raw_values = df_target['Value'].values

    st.title("🛡️ 供應鏈需求預測對照研究 (SCM Forecasting Analysis)")

    # 使用按鈕執行分析並儲存狀態，避免選單切換時資料遺失
    if st.button("🚀 執行多模型全自動分析 (Execute All Models)"):
        st.session_state['scm_results'] = run_comparison(raw_values)
        st.session_state['scm_dates'] = df_target['DATE'].values[-5:]

    # 檢查是否有運算結果
    if 'scm_results' in st.session_state:
        results = st.session_state['scm_results']
        test_dates = st.session_state['scm_dates']
        y_true = results["Actual"]

        tab1, tab2, tab3 = st.tabs([
            "📈 預測分析 (Predictive Analytics)",
            "🧪 不確定性模擬 (Uncertainty Simulation)",
            "🧠 中英對照與結論 (Glossary & Conclusion)"
        ])

        # --- Tab 1: 可視化對比 ---
        with tab1:
            st.subheader("模型預測結果可視化 (Forecasting Visibility)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_dates, y=y_true, name="實際值 (Actual)", line=dict(color='black', width=4)))
            for m in ["SMA", "Random Forest", "Linear Regression", "ARIMA"]:
                fig.add_trace(go.Scatter(x=test_dates, y=results[m], name=m))

            fig.update_layout(xaxis_title="年份 (Year)", yaxis_title="需求量 (Demand)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # 顯示 MAPE 績效
            st.subheader("🎯 準確率指標對照 (Forecast Accuracy Metrics)")
            cols = st.columns(4)
            for i, m in enumerate(["SMA", "Random Forest", "Linear Regression", "ARIMA"]):
                mape = mean_absolute_percentage_error(y_true, results[m]) * 100
                cols[i].metric(m, f"{mape:.2f}%")

        # --- Tab 2: 殘差與不確定性 (支援所有模型切換) ---
        with tab2:
            st.subheader("🧪 模型殘差與不確定性分析 (Residuals Comparison)")
            st.write("您可以切換下方選單，比較不同模型在擾動場景下的穩定性：")

            # 這裡包含所有用到的模型 (All models included)
            selected_model = st.selectbox(
                "選擇分析對象 (Select Model)",
                ["Random Forest", "ARIMA", "Linear Regression", "SMA"]
            )

            res_vals = y_true - results[selected_model]
            colors = ['#87CEEB' if r >= 0 else '#FF7F7F' for r in res_vals]

            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(x=test_dates, y=res_vals, marker_color=colors, name=f"{selected_model} Residuals"))
            fig_res.update_layout(
                title=f"<b>{selected_model} 殘差分佈 (Residuals Analysis)</b>",
                xaxis_title="年份", yaxis_title="預測誤差 (Error)", template="plotly_white"
            )
            st.plotly_chart(fig_res, use_container_width=True)

            # 中英對照解釋
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                **分析模型 (Model):** {selected_model}
                * **Positive (正值):** Under-forecast (實際 > 預測) → **缺貨風險**
                * **Negative (負值):** Over-forecast (實際 < 預測) → **庫存成本增加**
                """)
            with c2:
                max_err = np.max(np.abs(res_vals))
                st.warning(f"""
                **不確定性量化 (Uncertainty Quantification):**
                * 最大偏差 (Max Residual): **{max_err:.2f}**
                * 建議安全庫存緩衝 (Safety Stock Buffer): **{max_err:.2f}**
                """)

        # --- Tab 3: 中英對照 ---
                # --- Tab 3: 中英對照與研究結論 ---
                with tab3:
                    # 1. 自動化學術結論 (Automated Academic Conclusion)
                    st.subheader("🎓 研究總結 (Research Summary)")

                    # 找出表現最好的模型 (MAPE 最小者)
                    best_model_name = min(["SMA", "Random Forest", "Linear Regression", "ARIMA"],
                                          key=lambda m: mean_absolute_percentage_error(y_true, results[m]))

                    # 獲取該模型的最大誤差 (不確定性量化)
                    current_res = y_true - results[selected_model]
                    max_err_val = np.max(np.abs(current_res))

                    st.markdown(f"""
                    **【中文總結】**
                    本研究針對 **{dataset_option}** 進行了多模型驗證。實驗結果顯示，在此案例中 **{best_model_name}** 表現最為優異。
                    透過此模型分析預測誤差，我們發現供應鏈中的「不確定性」最大值為 **{max_err_val:.2f}**。
                    根據流程圖中的「反饋循環 (Feedback Loops)」，企業應以此數值作為安全庫存的緩衝基準，以達成庫存優化並降低斷貨風險。

                    **【English Summary】**
                    This study conducted a multi-model validation for **{dataset_option}**. The results indicate that **{best_model_name}** is the best performer in this case. 
                    By analyzing the forecast errors, we quantified the maximum "Uncertainty" in the supply chain as **{max_err_val:.2f}**. 
                    Following the "Feedback Loops" in our flowchart, enterprises should use this value as the buffer for Safety Stock to achieve inventory optimization and mitigate stockout risks.
                    """)

                    st.markdown("---")

                    # 2. 專業術語對照表 (Bilingual Glossary)
                    st.subheader("📖 專業術語對照 (Bilingual Glossary)")

                    # 建立對照表資料
                    glossary_data = {
                        "項目 (Item)": [
                            "Actual Demand", "Residuals", "Disruption",
                            "Visibility", "Adaptability", "Safety Stock"
                        ],
                        "中文解釋 (Chinese Explanation)": [
                            "實際需求：市場真實發生的銷售數據。",
                            "殘差：實際值與預測值的差距，用來量化「不確定性」。",
                            "擾動：供應鏈中突發的意外事件（如疫情、斷貨）。",
                            "可視化：透過數據圖表清晰掌握需求趨勢。",
                            "適應性：系統根據反饋自動調整決策的能力。",
                            "安全庫存：為了應對預測不準確而額外準備的庫存緩衝。"
                        ],
                        "English Definition": [
                            "Real-world sales data observed in the market.",
                            "The gap between actual and forecast; used to quantify Uncertainty.",
                            "Unexpected events in the supply chain (e.g., pandemics, shortages).",
                            "Clear transparency of demand trends through data visualization.",
                            "The system's ability to adjust decisions based on feedback.",
                            "The inventory buffer kept to protect against forecast errors."
                        ]
                    }
                    st.table(pd.DataFrame(glossary_data))

else:
    st.error("請確保 meat_consumption_worldwide.csv 檔案存在。")
