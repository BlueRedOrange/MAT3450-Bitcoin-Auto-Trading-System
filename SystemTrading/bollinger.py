# * What is 볼린저 strategy?
# 볼린저 밴드 전략은
# 주가의 변동성을 분석하기 위해 평균 이동선과 상한선, 하한선을 사용합니다.
# 상한선은 평균 이동선에 표준편차의 2배를 더한 값이고, 하한선은 평균 이동선에 표준편차의 2배를 뺀 값으로 계산됩니다.
# 주가가 하한선 아래로 내려가면 과매도로 판단해 매수 신호로 간주하며, 상한선을 넘어가면 과매수로 판단해 매도 신호로 간주합니다.
# 변동성이 큰 시장에서 유용하며, 반등 및 하락 시점을 예측하는 데 활용됩니다.

import pyupbit
import time
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from datetime import datetime

# 환경 변수 로드
load_dotenv()
access = os.environ["access"]  # access 키
secret = os.environ["secret"]  # secret 키
upbit = pyupbit.Upbit(access, secret)

# 분할 매수/매도 비율 설정
TRADE_RATIO = 0.75  # 부분 매매 X.
FILENAME = "live_trading_bollinger.xlsx"  # 거래 기록 파일 이름

# 매수 함수 정의
def buy_partial(ticker, total_balance, trade_ratio):

    """
       매수 주문을 실행하는 함수로, 지정된 자산(ticker)을 매수 비율(trade_ratio)에 따라 매수합니다.

       Args:
           ticker (str): 매수하려는 암호화폐 티커(예: "KRW-BTC").
           total_balance (float): 사용 가능한 총 잔고(KRW 단위).
           trade_ratio (float): 매수 비율 (0 ~ 1 범위, 예: 0.75).

       Returns:
           dict 또는 None: 매수 주문 정보(성공 시) 또는 None(실패 시).
    """

    # 매수 금액 계산: 총 잔고에서 trade_ratio 만큼의 비율을 곱한 값
    trade_amount = total_balance * trade_ratio
    # 업비트의 최소 거래 금액 조건 확인 (5,000 KRW 이상)
    if trade_amount >= 5000:  # 업비트의 최소 거래 금액 제한 (5000 KRW 이상)
        try:
            # 시장가 매수 주문 실행
            # 시장가 매수 주문 실행
            # upbit.buy_market_order: 업비트 API를 통해 현재 시장 가격으로 매수
            order = upbit.buy_market_order(ticker, trade_amount)
            # 매수 성공 시, 매수 금액 및 결과 출력
            print(f"[매수] {trade_amount:.2f} KRW 매수 완료!") # 매수 금액을 소수점 둘째 자리까지 출력
            return order # 매수 주문 결과 반환 (딕셔너리 형태)
        # 매수 중 오류 발생 시 예외 처리
        except Exception as e:
            # 오류 메시지를 출력하여 디버깅에 도움
            print(f"[매수 오류] {e}")
            # 매수 실패 시 None 반환
            return None
        # 최소 매수 금액 미달 시 사용자에게 실패 메시지 출력
    else:
        # 최소 금액 미달 시 실패 메시지 출력
        print("[매수 실패] 잔액 부족으로 매수하지 못했습니다.")
        return None # 매수 실패 시 None 반환

# 매도 함수 정의
def sell_partial(ticker, total_position, trade_ratio):

    """
       매도 주문을 실행하는 함수로, 지정된 자산(ticker)을 보유량(total_position)에서
       매도 비율(trade_ratio)에 따라 매도합니다.

       Args:
           ticker (str): 매도하려는 암호화폐 티커(예: "KRW-BTC").
           total_position (float): 현재 보유한 자산의 수량.
           trade_ratio (float): 매도 비율 (0 ~ 1 범위, 예: 0.75).

       Returns:
           dict 또는 None: 매도 주문 정보(성공 시) 또는 None(실패 시).
    """

    # 매도 수량 계산: 보유량(total_position)에 매도 비율(trade_ratio)을 곱한 값
    sell_amount = total_position * trade_ratio
    # 업비트의 최소 매도 수량 조건 확인 (0.0001 이상이어야 함)
    if sell_amount > 0.0001:
        try:
            # 시장가 매도 주문 실행
            # upbit.sell_market_order: 업비트 API를 사용하여 현재 시장 가격으로 매도
            order = upbit.sell_market_order(ticker, sell_amount)
            # 매도 성공 시 매도 수량 및 결과 출력
            print(f"[매도] {sell_amount:.6f} {ticker} 매도 완료!") # 매도 수량을 소수점 6자리까지 출력
            return order # 매도 주문 결과 반환 (딕셔너리 형태)
        # 매도 중 오류 발생 시 예외 처리
        except Exception as e:
            # 오류 메시지를 출력하여 디버깅에 도움
            print(f"[매도 오류] {e}")
            return None # 매도 실패 시 None 반환
    else:
        # 최소 매도 수량 미달 시 실패 메시지 출력
        print("[매도 실패] 보유량 부족으로 매도하지 못했습니다.")
        return None # 매도 실패 시 None 반환

# 주문 정보 조회 함수.
def get_order_info(uuid):

    """
       특정 주문 ID(uuid)에 대한 주문 정보를 조회하는 함수.

       Args:
           uuid (str): 업비트에서 생성된 주문의 고유 식별자(ID).

       Returns:
           dict 또는 None:
               - 주문 정보(딕셔너리 형태, 성공 시).
               - None(오류 발생 시).

       주요 기능:
           1. 업비트 API를 통해 주문 정보를 조회합니다.
           2. 주문의 상태, 체결 수량, 잔여 수량 등 상세한 정보를 반환합니다.
           3. 오류 발생 시 예외를 처리하고, None을 반환합니다.
    """

    try:
        # 주문 ID를 사용해 주문 정보를 조회
        # 주문 ID(uuid)를 사용하여 업비트 API의 get_order 메서드 호출
        order_info = upbit.get_order(uuid)
        # 정상적으로 주문 정보를 반환
        return order_info
    except Exception as e:
        # 조회 실패 시 오류 메시지 출력
        # 오류 발생 시 예외 처리 및 에러 메시지 출력
        print(f"[체결 정보 조회 오류] {e}")
        # 오류 시 None 반환
        return None

# 주문 체결 대기 함수
def wait_for_order_execution(uuid, max_retries=10, delay=1):

    """
        특정 주문 ID(uuid)의 체결 여부를 확인하고, 주문이 체결될 때까지 대기하는 함수.

        Args:
            uuid (str): 확인할 주문의 고유 ID.
            max_retries (int, optional): 체결 여부 확인을 위한 최대 재시도 횟수. 기본값은 10.
            delay (int, optional): 각 재시도 사이의 대기 시간(초). 기본값은 1초.

        Returns:
            dict 또는 None:
                - dict: 주문 정보(체결 성공 시).
                - None: 체결 실패(모든 재시도 후에도 체결되지 않은 경우).

        주요 기능:
            1. 주어진 주문 ID(uuid)에 대해 주문 정보를 주기적으로 조회합니다.
            2. 주문의 체결 여부를 확인하고, 체결될 때까지 최대 max_retries번 재시도합니다.
            3. 체결이 완료되면 주문 정보를 반환합니다.
            4. 체결되지 않았다면 실패 메시지를 출력하고 None을 반환합니다.
    """

    # 주문 체결 여부를 확인하기 위해 max_retries 횟수만큼 반복
    for _ in range(max_retries):
        # 주문 정보 조회 (get_order_info 함수 호출)
        order_info = get_order_info(uuid)

        # 주문 정보가 있고, 체결된 수량(executed_volume)이 0보다 크면 주문 체결로 간주
        if order_info and float(order_info.get('executed_volume', 0)) > 0:
            # 주문 체결 정보를 반환
            return order_info
        time.sleep(delay)  # 지정된 대기 시간(delay)만큼 대기 후 다시 시도
    # 최대 재시도 횟수 초과 시 체결 실패 메시지 출력
    print("[체결 대기 실패] 체결되지 않았거나 시간이 초과되었습니다.")
    return None  # 체결되지 않았음을 나타내기 위해 None 반환

def buy_partial_live(ticker, krw_balance, trade_ratio):

    """
        실시간 매수 주문을 실행하는 함수.
        주어진 자산(ticker)을 사용 가능한 KRW 잔고에서 매수 비율(trade_ratio)에 따라 매수합니다.
        주문이 체결될 때까지 대기하며, 체결 정보를 반환합니다.

        Args:
            ticker (str): 매수하려는 암호화폐의 티커(예: "KRW-BTC").
            krw_balance (float): 사용 가능한 KRW 잔고.
            trade_ratio (float): 매수 비율 (0~1 범위, 예: 0.75).

        Returns:
            dict 또는 None:
                - dict: 체결된 주문 정보(성공 시).
                - None: 주문 실패 또는 체결 실패 시.
    """
    # 매수 금액 계산 (KRW 잔고 * 매수 비율)
    trade_amount = krw_balance * trade_ratio

    # 업비트 최소 거래 금액 조건 확인 (5000 KRW 이상이어야 주문 가능)
    if trade_amount >= 5000:  # 최소 거래 금액
        # 시장가 매수 주문 실행
        order = upbit.buy_market_order(ticker, trade_amount)
        # 주문이 정상적으로 접수되고 UUID가 반환된 경우
        if order and "uuid" in order:
            print(f"[매수] {trade_amount:.2f} KRW 매수 완료! 주문 ID: {order['uuid']}")
            # 주문 체결 여부를 대기하며 확인
            order_info = wait_for_order_execution(order['uuid'])
            # 체결된 주문 정보가 있으면 반환
            if order_info and "executed_volume" in order_info:
                return order_info
        # 체결 정보가 없거나 오류 발생 시
        print("[매수 오류] 체결 정보 없음")
        return order  # 주문 객체 반환 (체결 실패 시)
    else:
        # 최소 금액 미달로 매수 실패
        print("[매수 실패] 잔액 부족으로 매수하지 못했습니다.")
        return None # 매수 실패 시 None 반환

def sell_partial_live(ticker, position, trade_ratio):

    """
       실시간 매도 주문을 실행하는 함수.
       보유한 자산(position)에서 매도 비율(trade_ratio)에 따라 시장가로 매도합니다.
       주문이 체결될 때까지 대기하며, 체결 정보를 반환합니다.

       Args:
           ticker (str): 매도하려는 암호화폐의 티커(예: "KRW-BTC").
           position (float): 현재 보유한 암호화폐의 수량.
           trade_ratio (float): 매도 비율 (0~1 범위, 예: 0.5).

       Returns:
           dict 또는 None:
               - dict: 체결된 주문 정보(성공 시).
               - None: 주문 실패 또는 체결 실패 시.
    """

    # 매도 수량 계산 (보유량 * 매도 비율)
    sell_amount = position * trade_ratio

    # 업비트 최소 거래 수량 조건 확인 (0.0001 이상이어야 주문 가능)
    if sell_amount > 0.0001:  # 최소 거래 수량 제한
        # 시장가 매도 주문 실행
        order = upbit.sell_market_order(ticker, sell_amount)
        # 주문이 정상적으로 접수되고 UUID가 반환된 경우
        if order and "uuid" in order:
            print(f"[매도] {sell_amount:.6f} {ticker} 매도 완료! 주문 ID: {order['uuid']}")
            # 주문 체결 여부를 대기하며 확인
            order_info = wait_for_order_execution(order['uuid'])
            # 체결된 주문 정보가 있으면 반환
            if order_info and "executed_volume" in order_info:
                return order_info
        # 체결 정보가 없거나 오류 발생 시
        print("[매수 오류] 체결 정보 없음")
        return order  # 주문 객체 반환 (체결 실패 시)
    else: # 최소 수량 미달로 매도 실패
        print("[매도 실패] 보유량 부족으로 매도하지 못했습니다.")
        return None # 매도 실패 시 None 반환

# 거래 기록 함수
def record_trade(date, action, price, amount, position, balance, filename=FILENAME):

    """
        거래 정보를 엑셀 파일에 기록하는 함수.
        새로운 거래 데이터를 추가하거나 기존 기록에 병합하여 저장합니다.

        Args:
            date (datetime): 거래가 발생한 날짜 및 시간.
            action (str): 거래 유형 (예: "Buy" 또는 "Sell").
            price (float): 거래 당시의 가격.
            amount (float): 거래된 금액 또는 수량.
            position (float): 거래 후의 총 보유량.
            balance (float): 거래 후의 잔여 잔고.
            filename (str, optional): 저장할 엑셀 파일 이름. 기본값은 `FILENAME`.

        동작 방식:
            1. 입력받은 거래 정보를 데이터프레임으로 생성합니다.
            2. 파일이 존재하지 않으면 새로운 엑셀 파일을 생성하여 데이터를 저장합니다.
            3. 파일이 이미 존재하면 기존 데이터를 읽어와 새로운 데이터를 병합한 뒤 저장합니다.
    """

    # 새로운 거래 데이터를 데이터프레임으로 생성
    trade_data = pd.DataFrame([{
        "Date": date,
        "Action": action,
        "Price": price,
        "Amount": amount,
        "Position": position,
        "Balance": balance
    }])

    # 지정된 파일이 존재하지 않는 경우 (최초 실행)
    if not os.path.exists(filename):
        # 새로운 엑셀 파일로 데이터 저장
        trade_data.to_excel(filename, index=False)
    else:
        # 기존 데이터를 읽어오고 새로운 데이터와 병합
        existing_data = pd.read_excel(filename) # 기존 데이터 로드
        updated_data = pd.concat([existing_data, trade_data], ignore_index=True) # 병합
        # 병합된 데이터를 엑셀 파일로 저장
        updated_data.to_excel(filename, index=False)


def get_bollinger_signal(ticker, interval):
    # pyupbit 라이브러리를 사용하여 특정 티커(ticker)와 봉 간격(interval)에 따라 데이터를 가져옵니다.
    # 여기서는 가장 최근 20개의 데이터를 기준으로 계산합니다.
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=20)

    # 볼린저 밴드의 중심선(middle): 최근 20개의 '종가(close)' 평균
    df['middle'] = df['close'].rolling(window=20).mean()

    # 상단선(upper): 중심선 + 최근 20개의 '종가(close)' 표준편차의 2배
    df['upper'] = df['middle'] + 2 * df['close'].rolling(window=20).std()

    # 하단선(lower): 중심선 - 최근 20개의 '종가(close)' 표준편차의 2배
    df['lower'] = df['middle'] - 2 * df['close'].rolling(window=20).std()

    # 함수는 가장 최신 시점의 상단선 값, 하단선 값, 종가를 반환합니다.
    return df['upper'].iloc[-1], df['lower'].iloc[-1], df['close'].iloc[-1]

# 벡테스팅 함수.
def backtest_bollinger(ticker, initial_balance=1000000, interval="minute5"):
    # pyupbit 라이브러리를 사용하여 지정된 티커와 간격(interval)으로 데이터 가져오기 (최근 2016개의 데이터)
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=2016)

    # 볼린저 밴드 계산
    # 중심선(middle): 최근 20개의 종가 평균
    df['middle'] = df['close'].rolling(window=20).mean()
    # 상단선(upper): 중심선 + 2배 표준편차
    df['upper'] = df['middle'] + 2 * df['close'].rolling(window=20).std()
    # 하단선(lower): 중심선 - 2배 표준편차
    df['lower'] = df['middle'] - 2 * df['close'].rolling(window=20).std()

    # 결측값(NaN) 제거 및 인덱스 초기화
    df = df.dropna().reset_index()

    # 초기 자산과 포지션 초기화
    balance = initial_balance  # 초기 잔고
    position = 0  # 초기 보유량
    trade_history = []  # 거래 내역 저장 리스트

    # 데이터프레임을 순회하며 백테스팅 실행
    for i in range(len(df)):
        # 현재 데이터의 종가, 상단선, 하단선 값 가져오기
        current_close = df['close'].iloc[i]
        current_upper = df['upper'].iloc[i]
        current_lower = df['lower'].iloc[i]
        current_date = df['index'].iloc[i]

        # 매수 조건: 종가가 하단선보다 낮고 잔고가 있을 때
        if current_close < current_lower and balance > 0:
            trade_amount = balance * TRADE_RATIO  # 거래 금액 계산
            position += trade_amount / current_close  # 보유량 증가
            balance -= trade_amount  # 잔고 감소
            # 거래 내역 기록
            trade_history.append([current_date, "Buy", current_close, trade_amount, position, balance])
            print(f"[매수] {current_date}, 가격: {current_close}, 잔고: {balance:.2f}, 보유량: {position:.6f}")

        # 매도 조건: 종가가 상단선보다 높고 보유량이 있을 때
        elif current_close > current_upper and position > 0:
            sell_amount = position * TRADE_RATIO  # 매도 수량 계산
            balance += sell_amount * current_close  # 잔고 증가
            position -= sell_amount  # 보유량 감소
            # 거래 내역 기록
            trade_history.append([current_date, "Sell", current_close, sell_amount * current_close, position, balance])
            print(f"[매도] {current_date}, 가격: {current_close}, 잔고: {balance:.2f}, 보유량: {position:.6f}")

    # 최종 잔고 및 수익률 계산
    final_balance = balance + (position * df['close'].iloc[-1])  # 잔고 + 현재 보유 자산의 총액
    total_return = (final_balance / initial_balance - 1) * 100  # 수익률 계산
    print(f"\n최종 잔고: {final_balance:.2f} KRW")
    print(f"총 수익률: {total_return:.2f}%")

    # 거래 기록을 엑셀 파일로 저장
    trade_df = pd.DataFrame(trade_history, columns=["Date", "Action", "Price", "Amount", "Position", "Balance"])
    trade_df.to_excel(f"backtest_bollinger_{interval}.xlsx", index=False)
    print(f"백테스팅 결과가 'backtest_bollinger_{interval}.xlsx'에 저장되었습니다.")

    # 백테스팅 결과 그래프 시각화
    plt.figure(figsize=(12, 8))

    # 종가 및 볼린저 밴드 그래프
    plt.plot(df['index'], df['close'], label="Close Price", color="blue", alpha=0.6)
    plt.plot(df['index'], df['middle'], label="Middle", color="orange")  # 중심선
    plt.plot(df['index'], df['upper'], label="Upper", color="green")  # 상단선
    plt.plot(df['index'], df['lower'], label="Lower", color="red")  # 하단선

    # 매수 및 매도 신호 표시
    plt.scatter(df[df['close'] < df['lower']]['index'], df[df['close'] < df['lower']]['close'], label="Buy Signal",
                marker="^", color="green")
    plt.scatter(df[df['close'] > df['upper']]['index'], df[df['close'] > df['upper']]['close'], label="Sell Signal",
                marker="v", color="red")

    # 그래프 제목 및 범례 추가
    plt.title(f"Bollinger Bands Backtest ({interval})")
    plt.legend()
    plt.show()


# 실시간 볼린저 밴드 기반 자동매매 함수
def bollinger_trading(ticker, interval="minute5"):
    print(f"실시간 볼린저 밴드 기반 자동매매 시작... ({interval} 기준)")
    while True:  # 무한 루프를 통해 실시간 트레이딩 수행
        try:
            # 볼린저 밴드 상한선, 하한선, 현재 종가 계산
            upper, lower, close = get_bollinger_signal(ticker, interval)

            # 현재 잔액(원화) 및 보유량(코인) 확인
            krw_balance = upbit.get_balance("KRW")  # 원화 잔액 확인
            position = upbit.get_balance(ticker)  # 코인 보유량 확인

            # NoneType 값 확인 및 기본값 설정
            # API 호출 실패나 잔액/보유량이 없는 경우 예외 처리
            if krw_balance is None:
                krw_balance = 0
            if position is None:
                position = 0

            # 매수 조건: 종가가 볼린저 하한선(lower) 아래로 내려갔을 때
            if close < lower and krw_balance > 5000:  # 최소 거래 금액(5000 KRW) 이상인지 확인
                print("[매수 신호] 볼린저 하한선 돌파")
                # 매수 실행
                order = buy_partial_live(ticker, krw_balance, TRADE_RATIO)
                if order:  # 주문 성공 시 거래 내역 기록
                    record_trade(datetime.now(), "Buy", close, krw_balance * TRADE_RATIO, position, krw_balance)

            # 매도 조건: 종가가 볼린저 상한선(upper) 위로 올라갔을 때
            elif close > upper and position > 0.0001:  # 최소 거래 수량(0.0001 코인) 이상인지 확인
                print("[매도 신호] 볼린저 상한선 돌파")
                # 매도 실행
                order = sell_partial_live(ticker, position, TRADE_RATIO)
                if order:  # 주문 성공 시 거래 내역 기록
                    record_trade(datetime.now(), "Sell", close, position * TRADE_RATIO, position, krw_balance)

            # 다음 데이터 확인 전 대기 시간 설정 (봉 기준에 따라 조정)
            time.sleep(300 if interval == "minute1" else 3600)  # 5분봉: 300초, 일봉: 3600초

        except Exception as e:  # 예외 발생 시 에러 메시지 출력 후 대기
            print(f"[에러] {e}")
            time.sleep(60)  # 60초 대기 후 재시도

# 프로그램 실행부
if __name__ == "__main__":
    # 티커와 초기 자산 설정
    TICKER = "KRW-BTC"  # 트레이딩할 자산 (여기서는 BTC)
    INITIAL_BALANCE = 1000000  # 초기 자산 설정 (100만 원)

    # 실행 모드 선택 (백테스팅 또는 실시간 트레이딩)
    mode = input("실행 모드를 선택하세요 (backtest / live): ").strip().lower()
    if mode == "backtest":  # 백테스팅 모드 선택 시
        # 봉 기준 입력 (1분, 5분, 또는 1일 기준)
        interval = input("봉 기준을 입력하세요 (minute1 / minute5 / day): ").strip().lower()
        if interval in ["minute1", "minute5", "day"]:  # 유효한 봉 기준 확인
            # 선택한 기준으로 백테스팅 함수 실행
            backtest_bollinger(TICKER, INITIAL_BALANCE, interval=interval)
        else:
            # 유효하지 않은 봉 기준 입력 시 경고 메시지 출력
            print("잘못된 입력입니다. 'minute1', 'minute5' 또는 'day'를 입력하세요.")

    elif mode == "live":  # 실시간 트레이딩 모드 선택 시
        # 봉 기준 입력 (5분 또는 1일 기준)
        interval = input("봉 기준을 입력하세요 (minute5 / day): ").strip().lower()
        if interval == "minute1":  # 1분봉 기준 실시간 트레이딩
            bollinger_trading(TICKER, interval="minute1")
        elif interval == "minute5":  # 5분봉 기준 실시간 트레이딩
            bollinger_trading(TICKER, interval="minute5")
        elif interval == "day":  # 일봉 기준 실시간 트레이딩
            bollinger_trading(TICKER, interval="day")
        else:
            # 유효하지 않은 봉 기준 입력 시 경고 메시지 출력
            print("잘못된 입력입니다. 'minute1 또는 'minute5' 또는 'day'를 입력하세요.")
    else:
        # 유효하지 않은 실행 모드 입력 시 경고 메시지 출력
        print("잘못된 입력입니다. 'backtest' 또는 'live'를 입력하세요.")
