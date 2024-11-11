import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
import ast
import modeling

# 파일 경로 설정
FILE_PATH = 'lotto_data.csv'

def get_latest_draw_number():
    url = "https://www.dhlottery.co.kr/gameResult.do?method=byWin"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        select_box = soup.find('select', {'id': 'dwrNoList'})
        if select_box:
            options = select_box.find_all('option')
            if options:
                return int(options[0]['value'])
        st.error("최신 회차 정보를 찾을 수 없습니다.")
        return None
    except requests.RequestException as e:
        st.error(f"웹사이트에 접근할 수 없습니다: {e}")
        return None
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        return None

def get_lotto_numbers(draw_number):
    url = f"https://www.dhlottery.co.kr/gameResult.do?method=byWin&drwNo={draw_number}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        win_numbers = soup.find_all('span', class_='ball_645')
        if not win_numbers:
            st.error(f"{draw_number}회차 당첨 번호를 찾을 수 없습니다.")
            return None

        numbers = [int(num.text) for num in win_numbers[:6]]
        bonus_number = int(win_numbers[6].text)

        table = soup.find('table', class_='tbl_data tbl_data_col')
        rows = table.find_all('tr')

        prize_info = []
        for row in rows[1:]:
            columns = row.find_all('td')
            rank = columns[0].text.strip()
            winners = columns[1].text.strip()
            prize_amount = columns[2].text.strip()
            
            winners_num = int(''.join(filter(str.isdigit, winners))) if winners != '0' else 0
            prize_amount_num = int(''.join(filter(str.isdigit, prize_amount))) if prize_amount != '0원' else 0
            
            winners_per_prize = 0 if prize_amount_num == 0 else winners_num / prize_amount_num

            prize_info.append({
                'rank': rank,
                'winners': winners,
                'prize_amount': prize_amount,
                'winners_per_prize': f"{winners_per_prize:.8f}"
            })

        draw_date_text = soup.find('p', class_='desc').text.split('(')[1].split(')')[0]
        draw_date = parse_draw_date(draw_date_text)

        return {
            'draw_number': draw_number,
            'numbers': numbers,
            'bonus_number': bonus_number,
            'prize_info': prize_info,
            'draw_date': draw_date
        }
    except Exception as e:
        st.error(f"{draw_number}회차 정보를 가져오는 중 오류 발생: {e}")
        return None

def parse_draw_date(date_text):
    date_text = date_text.replace(' 추첨', '')
    date_formats = ['%Y년 %m월 %d일', '%Y년']
    for date_format in date_formats:
        try:
            return datetime.strptime(date_text, date_format).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return date_text  # 형식이 맞지 않는 경우 원본 텍스트 반환

@st.cache_data
def load_or_create_data():
    if os.path.exists(FILE_PATH):
        df = pd.read_csv(FILE_PATH)
        df['numbers'] = df['numbers'].apply(ast.literal_eval)
        df['prize_info'] = df['prize_info'].apply(ast.literal_eval)
        
        # 날짜 형식 변환
        df['draw_date'] = df['draw_date'].apply(parse_draw_date)
        df['draw_date'] = pd.to_datetime(df['draw_date'])
        
        latest_saved_date = df['draw_date'].max()
        current_date = datetime.now()
        
        if (current_date - latest_saved_date).days > 7:
            latest_draw = get_latest_draw_number()
            if (latest_draw and latest_draw > df['draw_number'].max()):
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text(f"최신 데이터 업데이트 중... ({latest_draw}회차)")
                new_data = get_lotto_numbers(latest_draw)
                if new_data:
                    new_df = pd.DataFrame([new_data])
                    df = pd.concat([df, new_df], ignore_index=True)
                    df.to_csv(FILE_PATH, index=False)
                progress_bar.progress(100)
                status_text.text("데이터 업데이트 완료!")
    else:
        latest_draw = get_latest_draw_number()
        if latest_draw:
            data = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i in range(1, latest_draw + 1):
                status_text.text(f"{i}회차 정보 가져오는 중... ({i}/{latest_draw})")
                lotto_data = get_lotto_numbers(i)
                if lotto_data:
                    data.append(lotto_data)
                progress = int((i / latest_draw) * 100)
                progress_bar.progress(progress)
                
            if data:
                df = pd.DataFrame(data)
                df.to_csv(FILE_PATH, index=False)
                status_text.text("모든 데이터 로드 완료!")
            else:
                st.error("로또 데이터를 가져올 수 없습니다.")
                return pd.DataFrame()
        else:
            st.error("최신 회차 정보를 가져올 수 없습니다. 인터넷 연결을 확인해주세요.")
            return pd.DataFrame()
    
    return df

# Streamlit 앱 시작
st.title("복권 645 AI 번호 예측 프로그램:maded by Bryan Cho")

# 데이터 로드
st.write("로또 데이터를 로딩 중입니다. 잠시만 기다려주세요...")
df = load_or_create_data()

if df.empty:
    st.stop()

# 사이드바에 회차 선택 입력 추가
latest_draw = df['draw_number'].max()
selected_draw = st.sidebar.number_input("회차 선택", min_value=1, max_value=latest_draw, value=latest_draw)

# 선택된 회차의 로또 정보 가져오기
lotto_info = df[df['draw_number'] == selected_draw].iloc[0]

# 메인 페이지에 정보 표시
st.write(f"제 {lotto_info['draw_number']}회 로또645 당첨 정보 (추첨일: {lotto_info['draw_date']}):")

# 당첨 번호 시각화
st.write("## 당첨 번호")
numbers = lotto_info['numbers'] + [lotto_info['bonus_number']]
colors = ['#fbc400', '#69c8f2', '#ff7272', '#aaa', '#b0d840', '#c7c7c7', '#b0d840']

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
columns = [col1, col2, col3, col4, col5, col6, col7]

for num, color, col in zip(numbers, colors, columns):
    col.markdown(f'<div style="background-color:{color};border-radius:50%;width:50px;height:50px;display:flex;align-items:center;justify-content:center;margin:auto;"><span style="color:white;font-weight:bold;">{num}</span></div>', unsafe_allow_html=True)

# 당첨 정보 표시
st.write("## 당첨 내역")
prize_info = lotto_info['prize_info']
for prize in prize_info:
    st.write(f"{prize['rank']}: {prize['prize_amount']} (당첨자 수: {prize['winners']})")

# 데이터프레임으로 변환하여 표시
df_prize = pd.DataFrame(prize_info)
df_prize['winners_per_prize'] = df_prize['winners_per_prize'].astype(float)
st.write("## 당첨 정보 테이블")
st.dataframe(df_prize)

# AI 예측 섹션 추가
st.write("## AI 번호 예측")

if st.button("다음 회차 번호 예측"):
    with st.spinner("AI 모델 분석 중..."):
        # 데이터프레임에서 번호 추출
        numbers = np.array([row['numbers'] for _, row in df.iterrows()])
        
        # 모델 초기화 또는 업데이트
        modeling.initialize_or_update_model(numbers)
        
        # 다음 회차 번호 예측
        predicted_numbers = modeling.predict_next_numbers()
        
        st.write("AI가 예측한 다음 회차 번호:")
        cols = st.columns(6)
        for i, num in enumerate(predicted_numbers):
            cols[i].markdown(f'<div style="background-color:#1e90ff;border-radius:50%;width:50px;height:50px;display:flex;align-items:center;justify-content:center;margin:auto;"><span style="color:white;font-weight:bold;">{num}</span></div>', unsafe_allow_html=True)

    st.warning("주의: 이 예측은 과거 데이터를 기반으로 한 AI의 추측일 뿐입니다. 실제 당첨 번호와는 무관합니다.")