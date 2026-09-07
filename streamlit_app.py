from __future__ import annotations

import html
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import modeling
from lottery_data import LotteryDataError, refresh_lottery_data


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "lotto_data.csv"

st.set_page_config(
    page_title="Lotto Insight 645",
    page_icon="🍀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
        :root {
            --ink: #12231d;
            --muted: #65736d;
            --green: #0f7a52;
            --green-dark: #075c3c;
            --mint: #e9f7f1;
            --line: #dfe9e4;
            --cream: #fbfaf5;
            --gold: #f3b63f;
        }
        html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
        .stApp { background: linear-gradient(180deg, #f6fbf8 0, #ffffff 36rem); color: var(--ink); }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { background: #0b2b20; }
        [data-testid="stSidebar"] * { color: #eef8f3; }
        [data-testid="stSidebar"] .stButton button {
            background: rgba(255,255,255,.1); border-color: rgba(255,255,255,.25); color: white;
        }
        .hero {
            position: relative; overflow: hidden; padding: 2.2rem 2.4rem; border-radius: 28px;
            background: linear-gradient(125deg, #073d2a 0%, #0e704b 58%, #22a36f 100%);
            color: white; box-shadow: 0 18px 50px rgba(12, 91, 62, .18); margin-bottom: 1.4rem;
        }
        .hero:after {
            content: "645"; position: absolute; right: 1.6rem; top: -2.2rem; font-size: 10rem;
            font-weight: 800; color: rgba(255,255,255,.07); letter-spacing: -.08em;
        }
        .eyebrow { font-size: .78rem; letter-spacing: .13em; font-weight: 700; color: #a8ebcd; }
        .hero h1 { font-size: clamp(2rem, 5vw, 3.45rem); margin: .35rem 0 .55rem; letter-spacing: -.055em; }
        .hero p { max-width: 660px; margin: 0; color: #d9f3e7; line-height: 1.7; }
        .hero-badge {
            display: inline-block; margin-top: 1.25rem; padding: .45rem .75rem; border-radius: 999px;
            background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18); font-size: .78rem;
        }
        .section-head { margin: .65rem 0 1.05rem; }
        .section-head h2 { margin: 0; font-size: 1.55rem; letter-spacing: -.035em; }
        .section-head p { margin: .35rem 0 0; color: var(--muted); }
        .result-card, .pick-card {
            border: 1px solid var(--line); background: rgba(255,255,255,.9); border-radius: 22px;
            padding: 1.35rem 1.45rem; box-shadow: 0 8px 30px rgba(24, 63, 47, .055);
        }
        .pick-card { margin: .65rem 0; display: flex; align-items: center; gap: 1rem; }
        .pick-label { width: 56px; font-size: .8rem; font-weight: 700; color: var(--green); }
        .ball-row { display: flex; flex-wrap: wrap; align-items: center; gap: clamp(.38rem, 1.4vw, .75rem); }
        .lotto-ball {
            width: clamp(42px, 6vw, 56px); height: clamp(42px, 6vw, 56px); border-radius: 50%;
            display: inline-flex; align-items: center; justify-content: center; color: white;
            font-size: clamp(.9rem, 2vw, 1.08rem); font-weight: 800;
            box-shadow: inset 0 -5px 10px rgba(0,0,0,.14), 0 6px 14px rgba(16,43,33,.13);
        }
        .ball-1 { background: linear-gradient(145deg,#f6c84e,#e5a51f); }
        .ball-2 { background: linear-gradient(145deg,#65bced,#3688c3); }
        .ball-3 { background: linear-gradient(145deg,#f47c74,#d84f4a); }
        .ball-4 { background: linear-gradient(145deg,#969da4,#656d75); }
        .ball-5 { background: linear-gradient(145deg,#72bd72,#3b8f56); }
        .plus { color: #8a9690; font-size: 1.45rem; font-weight: 400; margin: 0 .05rem; }
        .bonus-wrap { position: relative; display: inline-flex; }
        .bonus-tag { position: absolute; top: -15px; width: 100%; text-align: center; color: var(--muted); font-size: .62rem; }
        .mini-note { color: var(--muted); font-size: .82rem; line-height: 1.6; }
        .status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; background:#55d596; }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.84); border: 1px solid var(--line); padding: 1rem 1.1rem;
            border-radius: 18px; box-shadow: 0 5px 18px rgba(24,63,47,.04);
        }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        .stButton > button {
            border: 0; border-radius: 12px; background: var(--green); color: white; font-weight: 700;
            min-height: 2.8rem; box-shadow: 0 7px 18px rgba(15,122,82,.18);
        }
        .stButton > button:hover { background: var(--green-dark); color: white; border: 0; }
        div[data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid var(--line); }
        button[data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: .65rem 1rem; }
        @media (max-width: 640px) {
            .block-container { padding: 1rem .9rem 3rem; }
            .hero { padding: 1.55rem 1.35rem; border-radius: 22px; }
            .hero:after { display:none; }
            .pick-card { align-items: flex-start; flex-direction: column; gap: .65rem; }
            .pick-label { width:auto; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def load_app_data() -> tuple[pd.DataFrame, object]:
    return refresh_lottery_data(DATA_PATH)


def ball_class(number: int) -> str:
    return f"ball-{min((int(number) - 1) // 10 + 1, 5)}"


def balls_html(numbers: list[int], bonus: int | None = None) -> str:
    balls = "".join(
        f'<span class="lotto-ball {ball_class(number)}">{int(number)}</span>'
        for number in numbers
    )
    if bonus is not None:
        balls += (
            '<span class="plus">+</span><span class="bonus-wrap">'
            '<span class="bonus-tag">보너스</span>'
            f'<span class="lotto-ball {ball_class(bonus)}">{int(bonus)}</span></span>'
        )
    return f'<div class="ball-row">{balls}</div>'


def currency(value: int) -> str:
    return f"{int(value):,}원"


def draw_patterns(numbers: list[int]) -> dict[str, str]:
    odd = sum(number % 2 for number in numbers)
    low = sum(number <= 22 for number in numbers)
    return {
        "번호 합계": str(sum(numbers)),
        "홀짝 균형": f"{odd} : {6 - odd}",
        "저·고 구간": f"{low} : {6 - low}",
    }


inject_styles()

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">LOTTO DATA LAB</div>
      <h1>Lotto Insight 645</h1>
      <p>공식 추첨 데이터를 한눈에 보고, 과거 흐름을 바탕으로 균형 잡힌 번호 조합을 가볍게 탐색하세요.</p>
      <span class="hero-badge"><span class="status-dot"></span>공식 데이터 · 로컬 안전 캐시</span>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("공식 추첨 데이터를 확인하고 있어요…"):
        df, data_status = load_app_data()
except LotteryDataError:
    st.error("저장된 데이터가 없고 공식 사이트에도 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.")
    st.stop()

if df.empty:
    st.error("표시할 추첨 데이터가 없습니다.")
    st.stop()

latest_draw = int(df["draw_number"].max())
latest_info = df[df["draw_number"] == latest_draw].iloc[0]

with st.sidebar:
    st.markdown("### Lotto Insight")
    st.caption("데이터 상태")
    if data_status.online:
        st.success(data_status.message)
    else:
        st.warning(data_status.message)
    st.markdown(
        f"**마지막 데이터**  \n{latest_draw}회 · {latest_info['draw_date']:%Y.%m.%d}"
    )
    if st.button("데이터 새로고침", use_container_width=True):
        load_app_data.clear()
        st.rerun()
    st.divider()
    st.caption("데이터 출처")
    st.markdown("[동행복권 공식 사이트](https://www.dhlottery.co.kr/lt645/result)")
    st.caption("데이터는 30분 단위로 확인하며, 연결 장애 시 마지막 정상 데이터를 표시합니다.")

if not data_status.online:
    st.warning(
        f"공식 사이트 응답이 지연 중입니다. 서비스는 중단하지 않고 저장된 {latest_draw}회 데이터로 동작합니다."
    )

result_tab, recommend_tab, stats_tab = st.tabs(
    ["🎯 당첨 결과", "✨ 데이터 추천", "📊 흐름 분석"]
)

with result_tab:
    st.markdown(
        '<div class="section-head"><h2>회차별 당첨 결과</h2><p>공식 당첨번호와 등위별 정보를 확인하세요.</p></div>',
        unsafe_allow_html=True,
    )
    draw_options = df["draw_number"].astype(int).sort_values(ascending=False).tolist()
    selected_draw = st.selectbox(
        "조회 회차",
        draw_options,
        format_func=lambda draw: f"제 {draw:,}회",
        label_visibility="collapsed",
    )
    lotto_info = df[df["draw_number"] == selected_draw].iloc[0]
    selected_numbers = [int(number) for number in lotto_info["numbers"]]
    prize_info = lotto_info["prize_info"]

    st.markdown(
        f"""
        <div class="result-card">
          <div class="eyebrow" style="color:#0f7a52">{int(selected_draw):,}회 · {lotto_info['draw_date']:%Y년 %m월 %d일}</div>
          <h3 style="margin:.45rem 0 1.35rem; font-size:1.3rem">당첨번호</h3>
          {balls_html(selected_numbers, int(lotto_info['bonus_number']))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    first_prize = prize_info[0] if prize_info else {"prize_amount": 0, "winners": 0}
    metrics = st.columns(4)
    metrics[0].metric("1등 당첨금", currency(first_prize["prize_amount"]))
    metrics[1].metric("1등 당첨자", f"{int(first_prize['winners']):,}명")
    for column, (label, value) in zip(metrics[2:], draw_patterns(selected_numbers).items()):
        column.metric(label, value)

    st.markdown("#### 등위별 당첨 정보")
    prize_table = pd.DataFrame(prize_info).rename(
        columns={"rank": "등위", "winners": "당첨자 수", "prize_amount": "1인당 당첨금"}
    )
    if not prize_table.empty:
        prize_table["당첨자 수"] = prize_table["당첨자 수"].map(lambda value: f"{int(value):,}명")
        prize_table["1인당 당첨금"] = prize_table["1인당 당첨금"].map(currency)
        st.dataframe(prize_table, hide_index=True, use_container_width=True)

with recommend_tab:
    st.markdown(
        '<div class="section-head"><h2>데이터 기반 번호 추천</h2><p>최근 흐름·장기 빈도·미출현 간격을 조합해 다양한 균형 조합을 만듭니다.</p></div>',
        unsafe_allow_html=True,
    )
    control_a, control_b, control_c = st.columns([1, 1, 1.25])
    set_count = control_a.select_slider("추천 게임 수", options=list(range(1, 11)), value=5)
    max_lookback = min(300, len(df))
    lookback_options = sorted(set([value for value in (30, 50, 80, 120, 200, 300) if value <= max_lookback] + [max_lookback]))
    lookback = control_b.select_slider("분석 회차", options=lookback_options, value=min(80, max_lookback))
    regenerate = control_c.button("새 조합 만들기", use_container_width=True)

    session_key = f"recommendations_{set_count}_{lookback}_{latest_draw}"
    if regenerate or session_key not in st.session_state:
        seed = time.time_ns() if regenerate else latest_draw
        st.session_state[session_key] = modeling.generate_recommendations(
            df["numbers"].tolist(), count=set_count, lookback=lookback, seed=seed
        )

    for index, numbers in enumerate(st.session_state[session_key], start=1):
        stats = draw_patterns(numbers)
        escaped_summary = html.escape(
            f"합계 {stats['번호 합계']} · 홀짝 {stats['홀짝 균형']} · 저고 {stats['저·고 구간']}"
        )
        st.markdown(
            f"""
            <div class="pick-card">
              <div class="pick-label">GAME {index}</div>
              <div style="flex:1">{balls_html(numbers)}</div>
              <div class="mini-note">{escaped_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "모든 로또 조합의 1등 확률은 동일합니다. 추천은 과거 데이터를 활용한 탐색 기능이며, 당첨 가능성을 높인다는 의미가 아닙니다."
    )

with stats_tab:
    st.markdown(
        '<div class="section-head"><h2>번호 흐름 분석</h2><p>선택한 기간의 출현 빈도와 조합 특성을 탐색하세요.</p></div>',
        unsafe_allow_html=True,
    )
    analysis_max = min(300, len(df))
    analysis_window = st.slider(
        "분석 기간", min_value=20, max_value=analysis_max, value=min(100, analysis_max), step=10
    )
    recent_draws = df.tail(analysis_window)
    flattened = np.concatenate(recent_draws["numbers"].to_numpy())
    frequencies = np.bincount(flattened.astype(int), minlength=46)[1:]
    frequency_df = pd.DataFrame(
        {"번호": np.arange(1, 46), "출현 횟수": frequencies}
    ).set_index("번호")
    hot_numbers, cold_numbers = modeling.describe_trends(df["numbers"].tolist(), analysis_window)

    hot_column, cold_column = st.columns(2)
    with hot_column:
        st.markdown("##### 최근 가중치 상위")
        st.markdown(balls_html(hot_numbers), unsafe_allow_html=True)
        st.caption("최근 출현에 더 큰 가중치를 둔 탐색 지표입니다.")
    with cold_column:
        st.markdown("##### 최근 가중치 하위")
        st.markdown(balls_html(cold_numbers), unsafe_allow_html=True)
        st.caption("낮은 가중치가 다음 회차 미출현을 뜻하지는 않습니다.")

    st.markdown("##### 번호별 출현 횟수")
    st.bar_chart(frequency_df, height=360)

    recent_sums = recent_draws["numbers"].apply(sum)
    recent_odds = recent_draws["numbers"].apply(lambda row: sum(number % 2 for number in row))
    stat_metrics = st.columns(3)
    stat_metrics[0].metric("평균 번호 합", f"{recent_sums.mean():.1f}")
    stat_metrics[1].metric("가장 잦은 홀수 개수", f"{int(recent_odds.mode().iloc[0])}개")
    stat_metrics[2].metric("회차당 번호", "6개 / 45개")

st.divider()
st.caption(
    "Lotto Insight 645 · 데이터 출처: 동행복권 · 본 서비스는 정보와 오락을 위한 도구이며 구매를 권유하지 않습니다."
)
