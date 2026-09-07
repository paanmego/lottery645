# Lotto Insight 645

동행복권 공식 추첨 결과를 안정적으로 동기화하고, 과거 데이터 흐름을 바탕으로 균형 잡힌 번호 조합을 탐색하는 Streamlit 대시보드입니다.

## 주요 기능

- 공식 최신 JSON 경로와 10회차 배치 동기화
- 연결/응답 타임아웃, 자동 재시도, 로컬 캐시 폴백
- 회차별 당첨번호·등위별 당첨금 확인
- 최근 빈도·장기 빈도·미출현 간격 기반 경량 추천
- 모바일 대응 UI, 번호 빈도와 패턴 분석
- API 파싱·장애 처리·추천 유효성 자동 테스트

> 로또 추첨은 독립적인 무작위 사건이며 모든 조합의 당첨 확률은 같습니다. 추천 기능은 정보와 오락을 위한 탐색 도구입니다.

## 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501`을 엽니다.

## 데이터 갱신

```bash
python scripts/sync_data.py
```

앱은 30분 단위로 공식 데이터를 확인합니다. 공식 사이트가 응답하지 않으면 오류 원문을 사용자에게 노출하거나 서비스를 중단하지 않고 마지막 정상 CSV를 사용합니다.

## 테스트

```bash
pip install -r requirements-dev.txt
pytest -q
```

자세한 적용 단계와 후속 계획은 [UPDATE_PLAN.md](UPDATE_PLAN.md)를 참고하세요.
