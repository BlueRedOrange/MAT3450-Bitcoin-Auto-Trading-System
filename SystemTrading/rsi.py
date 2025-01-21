# * What is RSI strategy?
# RSI (Relative Strength Index) 전략은
# 특정 기간 동안 가격 상승과 하락의 비율을 기반으로
# 시장의 과매수와 과매도 상태를 평가하는 전략입니다.
# RSI 값이 30 이하일 때 과매도 상태로 간주하여 매수 신호를,
# 70 이상일 때 과매수 상태로 간주하여 매도 신호를 제공합니다.
# 이 전략은 시장의 반전을 예상하며, 가격이 과도하게 하락하거나 상승했을 때
# 진입 또는 청산 결정을 돕습니다. 시장의 변동성을 활용한 단기 매매에 적합합니다.

# 필요한 라이브러리 임포트
import pyupbit  # 업비트 API를 사용하여 거래 데이터 조회 및 주문 실행
import time  # 시간 지연 및 반복 제어
import pandas as pd  # 데이터 처리를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화 라이브러리
from dotenv import load_dotenv  # 환경 변수를 로드하기 위한 라이브러리
import os  # 파일 및 환경 변수 관련 작업을 위한 라이브러리
from datetime import datetime  # 날짜 및 시간 처리를 위한 라이브러리

# 환경 변수 로드 및 API 키 설정
load_dotenv()  # .env 파일에서 환경 변수를 로드
access = os.environ["access"]  # 업비트 API 액세스 키
secret = os.environ["secret"]  # 업비트 API 시크릿 키
upbit = pyupbit.Upbit(access, secret)  # 업비트 객체 초기화

# 설정
TRADE_RATIO = 0.25  # 한 번에 매수/매도 비율 (75%)
FEE = 0.0005  # 거래 수수료 비율
FILENAME = "live_trading_rsi.xlsx"  # 거래 기록 파일 이름

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


# RSI 계산 함수
def calculate_rsi(df, period=14):
    # 1. 가격 변화 계산: 각 시점의 종가 차이를 계산 (현재 종가 - 이전 종가)
    delta = df['close'].diff()

    # 2. 상승 및 하락만 분리
    # 상승분: 가격 상승분(delta > 0)을 유지하고 나머지는 0으로 대체
    gain = delta.where(delta > 0, 0)
    # 하락분: 가격 하락분(delta < 0)의 절댓값을 유지하고 나머지는 0으로 대체
    loss = -delta.where(delta < 0, 0)

    # 3. 평균 상승분 및 평균 하락분 계산 (기간: 14일이 기본값)
    # 최근 period(14) 기간 동안의 평균 상승분을 계산
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    # 최근 period(14) 기간 동안의 평균 하락분을 계산
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # 4. 상대 강도(RS) 계산: 평균 상승분 / 평균 하락분
    rs = avg_gain / avg_loss

    # 5. RSI 계산: 100 - (100 / (1 + RS))
    # RSI는 0에서 100 사이의 값으로 표현되며, 값이 높을수록 과매수, 낮을수록 과매도로 간주
    rsi = 100 - (100 / (1 + rs))

    # 최종적으로 RSI 값을 반환
    return rsi


# RSI 기반 백테스팅 함수
def backtest_rsi(ticker, initial_balance=1000000, rsi_buy=30, rsi_sell=80, trade_ratio=0.5, interval="minute5"):
    # 1. OHLCV 데이터 가져오기: pyupbit 라이브러리를 사용하여 주어진 간격(interval)과 과거 데이터 개수(count)를 가져옴
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=2016)

    # 2. RSI 계산: 'calculate_rsi' 함수를 사용하여 데이터프레임에 RSI 값을 추가
    df['RSI'] = calculate_rsi(df)

    # 3. 결측치 제거: RSI 계산 시 NaN 값이 포함되므로 이를 제거하고 인덱스를 재정렬
    df = df.dropna().reset_index()

    # 초기 잔고 및 보유량 설정
    balance = initial_balance  # 시작 잔고
    position = 0  # 초기 보유량 (매수 전 0)

    # 거래 기록 저장을 위한 리스트 초기화
    trade_history = []

    # 4. 데이터 순회: 각 행에서 RSI 및 가격 정보를 사용해 매수/매도 조건을 평가
    for i in range(len(df)):
        current_close = df['close'].iloc[i]  # 현재 종가
        current_rsi = df['RSI'].iloc[i]  # 현재 RSI 값
        current_date = df['index'].iloc[i]  # 현재 날짜

        # 4.1 매수 조건: RSI가 'rsi_buy' 이하이며 잔고가 있을 때
        if current_rsi < rsi_buy and balance > 0:
            trade_amount = balance * trade_ratio  # 매수 금액은 잔고의 일정 비율
            # 매수 후 보유량 업데이트 (수수료 고려)
            position += trade_amount / (current_close * (1 + FEE))
            # 매수 후 잔고 감소
            balance -= trade_amount
            # 거래 기록 추가
            trade_history.append([current_date, "Buy", current_close, trade_amount, position, balance])
            print(f"[매수] {current_date}, 가격: {current_close}, 잔고: {balance:.2f}, 보유량: {position:.6f}")

        # 4.2 매도 조건: RSI가 'rsi_sell' 이상이며 보유량이 있을 때
        elif current_rsi > rsi_sell and position > 0:
            sell_amount = position * trade_ratio  # 매도 수량은 보유량의 일정 비율
            # 매도 후 잔고 업데이트 (수수료 고려)
            balance += sell_amount * current_close * (1 - FEE)
            # 매도 후 보유량 감소
            position -= sell_amount
            # 거래 기록 추가
            trade_history.append([current_date, "Sell", current_close, sell_amount * current_close, position, balance])
            print(f"[매도] {current_date}, 가격: {current_close}, 잔고: {balance:.2f}, 보유량: {position:.6f}")

    # 5. 최종 잔고 및 수익률 계산
    # 모든 데이터 순회 후 잔고와 보유량 기반으로 최종 잔고 및 수익률 계산
    final_balance = balance + (position * df['close'].iloc[-1])  # 보유량의 현재 가치 포함
    total_return = (final_balance / initial_balance - 1) * 100
    print(f"\n최종 잔고: {final_balance:.2f} KRW")
    print(f"총 수익률: {total_return:.2f}%")

    # 6. 거래 기록 엑셀 파일로 저장
    trade_df = pd.DataFrame(trade_history, columns=["Date", "Action", "Price", "Amount", "Position", "Balance"])
    trade_df.to_excel(f"backtest_rsi_{interval}.xlsx", index=False)
    print(f"백테스팅 결과가 'backtest_rsi_{interval}.xlsx'에 저장되었습니다.")

    # 7. 결과 시각화
    plt.figure(figsize=(14, 10))

    # 7.1 종가와 매수/매도 신호 그래프
    plt.subplot(2, 1, 1)
    plt.plot(df['index'], df['close'], label="Close Price", color="blue", alpha=0.6)
    plt.scatter(df[df['RSI'] < rsi_buy]['index'], df[df['RSI'] < rsi_buy]['close'], label="Buy Signal", marker="^",
                color="green")
    plt.scatter(df[df['RSI'] > rsi_sell]['index'], df[df['RSI'] > rsi_sell]['close'], label="Sell Signal", marker="v",
                color="red")
    plt.title(f"RSI Backtest ({interval}): Close Price with Buy/Sell Signals")
    plt.legend()

    # 7.2 RSI와 매수/매도 기준선 그래프
    plt.subplot(2, 1, 2)
    plt.plot(df['index'], df['RSI'], label="RSI", color="purple")
    plt.axhline(rsi_buy, color="green", linestyle="--", label=f"RSI Buy ({rsi_buy})")
    plt.axhline(rsi_sell, color="red", linestyle="--", label=f"RSI Sell ({rsi_sell})")
    plt.title(f"RSI Indicator with Buy/Sell Zones ({interval})")
    plt.legend()
    plt.show()


# 실시간 RSI 기반 트레이딩 함수
def rsi_trading(ticker, rsi_buy=30, rsi_sell=80, trade_ratio=0.5, interval="minute5"):
    print(f"실시간 RSI 트레이딩 시작... ({interval} 기준)")  # 트레이딩 시작 메시지 출력
    while True:
        try:
            # 1. 데이터 가져오기
            # 봉 간격(interval)에 맞춰 지정된 개수(14개의 데이터)를 가져옴
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=14)

            # 2. RSI 계산
            # 최신 RSI 값을 계산하여 가져옴
            current_rsi = calculate_rsi(df).iloc[-1]

            # 3. 현재 종가 확인
            # 데이터에서 가장 최근 '종가'를 가져옴
            current_price = df['close'].iloc[-1]

            # 4. 잔고 조회
            # 현재 보유한 원화(KRW) 잔액
            krw_balance = upbit.get_balance("KRW") or 0
            # 현재 보유한 코인 잔량
            position = upbit.get_balance(ticker) or 0

            # 5. 현재 상태 출력
            # 현재 시각, RSI 값, 현재 종가를 출력
            print(f"[{datetime.now()}] RSI: {current_rsi:.2f}, 현재 가격: {current_price:.2f}")

            # 6. 매수 조건 확인
            if current_rsi < rsi_buy and krw_balance > 5000:  # RSI가 매수 기준보다 낮고 원화 잔고가 충분할 때
                # 매수 주문 실행
                order = buy_partial_live(ticker, krw_balance, trade_ratio)
                if order:
                    # 매수 거래 기록 저장
                    record_trade(datetime.now(), "Buy", current_price, order['executed_volume'], position, krw_balance)

            # 7. 매도 조건 확인
            elif current_rsi > rsi_sell and position > 0.0001:  # RSI가 매도 기준보다 높고 보유량이 충분할 때
                # 매도 주문 실행
                order = sell_partial_live(ticker, position, trade_ratio)
                if order:
                    # 매도 거래 기록 저장
                    record_trade(datetime.now(), "Sell", current_price, order['executed_volume'], position, krw_balance)

            # 8. 대기 시간 설정
            # 봉 간격(interval)에 따라 대기 시간 설정 (5분봉이면 300초, 일봉이면 3600초)
            time.sleep(300 if interval == "minute1" else 3600)

        except Exception as e:
            # 예외 발생 시 에러 메시지 출력
            print(f"[에러] {e}")
            # 에러 발생 시 60초 대기 후 재시도
            time.sleep(60)


# 실행 블록
if __name__ == "__main__":
    TICKER = "KRW-BTC"  # 거래 대상 코인
    INITIAL_BALANCE = 1000000  # 초기 잔고 설정

    # 실행 모드 선택
    mode = input("실행 모드를 선택하세요 (backtest / live): ").strip().lower()
    if mode == "backtest":
        # 백테스팅 모드: 백테스팅 수행
        interval = input("봉 기준을 입력하세요 (minute1 / minute5 / day): ").strip().lower()
        if interval in ["minute1", "minute5", "day"]:
            # 올바른 봉 기준 입력 시 백테스팅 실행
            backtest_rsi(TICKER, INITIAL_BALANCE, interval=interval)
        else:
            # 잘못된 입력 처리
            print("잘못된 입력입니다. 'minute1', 'minute5' 또는 'day'를 입력하세요.")
    elif mode == "live":
        # 실시간 트레이딩 모드
        interval = input("봉 기준을 입력하세요 (minute5 / day): ").strip().lower()
        if interval == "minute1":
            # 1분봉 실시간 트레이딩 실행
            rsi_trading(TICKER, interval="minute1")
        elif interval == "minute5":
            # 5분봉 실시간 트레이딩 실행
            rsi_trading(TICKER, interval="minute5")
        elif interval == "day":
            # 일봉 실시간 트레이딩 실행
            rsi_trading(TICKER, interval="day")
        else:
            # 잘못된 입력 처리
            print("잘못된 입력입니다. 'minute1' 또는 'minute5' 또는 'day'를 입력하세요.")
    else:
        # 잘못된 모드 입력 처리
        print("잘못된 입력입니다. 'backtest' 또는 'live'를 입력하세요.")
