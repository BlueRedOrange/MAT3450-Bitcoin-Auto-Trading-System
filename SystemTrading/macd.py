# * What is macd strategy?
# MACD 전략은 단기 이동평균(12일)과 장기 이동평균(26일)의 차이를 계산한 MACD 선과,
# 이 MACD 선의 9일 지수 이동평균인 신호선을 비교하여
# 매수/매도 신호를 포착하는 전략입니다. MACD가 신호선 위로 상승하면 매수 신호,
# 아래로 하락하면 매도 신호로 간주합니다. 이 전략은 추세의 방향과 강도를 파악하는 데 유용하며,
# 강한 추세를 따르는 시장에서 효과적입니다.

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

# 전역 변수 설정
TRADE_RATIO = 0.75  # 매수/매도 시 자산의 75%를 사용, 즉, 분할 매수/매도 비율 지정.
FILENAME = "live_trading_macd.xlsx"  # 거래 기록 저장 파일 이름
THRESHOLD = 0.001  # MACD와 신호선 간 차이에 따라 매수/매도 신호를 발생시키는 임계값# 매수 함수 정의

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

# MACD 계산 함수
def get_macd(df):
    # df['close'] 열의 데이터로 12일 지수 이동 평균 (EMA) 계산
    ema12 = df['close'].ewm(span=12, adjust=False).mean()

    # df['close'] 열의 데이터로 26일 지수 이동 평균 (EMA) 계산
    ema26 = df['close'].ewm(span=26, adjust=False).mean()

    # MACD 값 계산: 12일 EMA - 26일 EMA
    df['MACD'] = ema12 - ema26

    # MACD 값의 9일 지수 이동 평균 계산: Signal Line
    df['macd_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 계산된 DataFrame 반환
    return df


# 실시간 MACD 거래 함수
def macd_trading(ticker, interval="minute5"):
    """
    실시간으로 MACD를 사용하여 매수/매도 신호를 감지하고 자동으로 거래를 수행하는 함수입니다.

    Args:
        ticker (str): 거래할 종목 코드 (예: "KRW-BTC").
        interval (str): 데이터 시간 간격 (예: "minute1", "minute5", "day").
    """
    # 함수 시작 메시지 출력
    print(f"실시간 MACD 트레이딩 시작... ({interval})")

    # 무한 루프: 실시간 거래를 계속 수행
    while True:
        try:
            # 1. 현재 시장 데이터를 PyUpbit API를 통해 가져옴
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=2016)  # 최근 2016개의 캔들 데이터 요청

            # 데이터 가져오기 실패 시 경고 메시지를 출력하고, 60초 대기 후 재시도
            if df is None:
                print("[경고] 데이터를 가져오지 못했습니다.")
                time.sleep(60)
                continue

            # 2. MACD 및 Signal Line 계산
            df = get_macd(df)  # get_macd 함수 호출
            macd = df['MACD'].iloc[-1]  # 최신 MACD 값
            signal = df['macd_signal'].iloc[-1]  # 최신 Signal Line 값
            current_price = df['close'].iloc[-1]  # 현재 종가

            # 3. 잔액 정보 가져오기
            krw_balance = upbit.get_balance("KRW") or 0  # 보유 KRW 잔액
            ticker_balance = upbit.get_balance(ticker) or 0  # 보유 암호화폐 잔량

            # 4. 상태 출력: 현재 시간, MACD, Signal Line, 현재 가격
            print(f"[{datetime.now()}] MACD: {macd:.6f}, Signal: {signal:.6f}, Current Price: {current_price:.2f}")

            # 5. 매수 조건
            # MACD 값이 Signal Line을 Threshold만큼 초과한 경우 + KRW 잔액이 5000원 이상
            if (macd - signal) > THRESHOLD and krw_balance > 5000:
                # 시장가 매수 주문 실행
                order = buy_partial(ticker, krw_balance, TRADE_RATIO)
                if order:  # 주문 성공 시 거래 기록 저장
                    record_trade(
                        datetime.now(), "Buy", current_price,
                        krw_balance * TRADE_RATIO, ticker_balance, krw_balance
                    )

            # 6. 매도 조건
            # Signal Line이 MACD 값을 Threshold만큼 초과한 경우 + 보유 암호화폐 잔량이 0.0001 이상
            elif (signal - macd) > THRESHOLD and ticker_balance > 0.0001:
                # 시장가 매도 주문 실행
                order = sell_partial(ticker, ticker_balance, TRADE_RATIO)
                if order:  # 주문 성공 시 거래 기록 저장
                    record_trade(
                        datetime.now(), "Sell", current_price,
                        ticker_balance * TRADE_RATIO, ticker_balance, krw_balance
                    )

            # 7. 다음 데이터 확인 전 대기
            # "minute1"인 경우 5분 대기, 그 외 시간 간격은 1시간 대기
            time.sleep(300 if interval == "minute1" else 3600)

        except Exception as e:
            # 예외 발생 시 에러 메시지 출력 및 60초 대기
            print(f"에러 발생: {e}")
            time.sleep(60)


# MACD 백테스팅 함수
def backtest_macd(ticker, initial_balance=1000000, interval="5minute"):
    """
    MACD 지표를 기반으로 매수/매도 전략을 백테스팅하는 함수.

    Args:
        ticker (str): 거래할 종목 코드 (예: "KRW-BTC").
        initial_balance (int): 초기 투자 금액.
        interval (str): 데이터 시간 간격 (예: "minute1", "minute5", "day").
    """
    # 1. 과거 데이터 가져오기
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=1000)  # 최대 1000개의 데이터 요청
    if df is None:  # 데이터 가져오기 실패 시 오류 메시지 출력 후 종료
        print("[오류] 데이터를 가져오지 못했습니다.")
        return

    # 2. MACD 및 Signal Line 계산
    df = get_macd(df)  # MACD 계산 함수 호출
    df['Signal'] = 0  # 매수/매도 신호 초기화
    # 매수 신호 조건: MACD > Signal Line + THRESHOLD
    df.loc[df['MACD'] > (df['macd_signal'] + THRESHOLD), 'Signal'] = 1
    # 매도 신호 조건: MACD < Signal Line - THRESHOLD
    df.loc[df['MACD'] < (df['macd_signal'] - THRESHOLD), 'Signal'] = -1

    # 3. 초기 상태 설정
    position = 0  # 보유 암호화폐 수량
    balance = initial_balance  # 초기 잔고
    trade_history = []  # 거래 내역 기록

    print("백테스팅 시작...")

    # 4. 데이터 순회하며 매수/매도 실행
    for i in range(len(df)):
        current_close = df['close'].iloc[i]  # 현재 종가
        current_signal = df['Signal'].iloc[i]  # 현재 매수/매도 신호
        current_date = df.index[i]  # 현재 날짜

        # 매수 조건: 매수 신호가 발생하고 잔고가 남아 있는 경우
        if current_signal == 1 and balance > 0:
            trade_amount = balance * TRADE_RATIO  # 매수 금액 = 잔고 * TRADE_RATIO
            position += trade_amount / current_close  # 암호화폐 수량 증가
            balance -= trade_amount  # 잔고 감소
            trade_history.append([current_date, "Buy", current_close, trade_amount, position, balance])  # 거래 내역 기록
            # 콘솔 출력
            print(f"[매수] Date: {current_date}, Price: {current_close:.2f}, "
                  f"Amount: {trade_amount:.2f}, Position: {position:.6f}, Balance: {balance:.2f}")

        # 매도 조건: 매도 신호가 발생하고 암호화폐를 보유한 경우
        elif current_signal == -1 and position > 0:
            sell_amount = position * TRADE_RATIO  # 매도 수량 = 보유량 * TRADE_RATIO
            balance += sell_amount * current_close  # 잔고 증가
            position -= sell_amount  # 보유량 감소
            trade_history.append(
                [current_date, "Sell", current_close, sell_amount * current_close, position, balance])  # 거래 내역 기록
            # 콘솔 출력
            print(f"[매도] Date: {current_date}, Price: {current_close:.2f}, "
                  f"Amount: {sell_amount * current_close:.2f}, Position: {position:.6f}, Balance: {balance:.2f}")

    # 5. 최종 잔고 및 수익률 계산
    final_balance = balance + (position * df['close'].iloc[-1])  # 잔고 + 보유량 * 마지막 종가
    total_return = (final_balance / initial_balance - 1) * 100  # 총 수익률 계산
    print(f"\n최종 잔고: {final_balance:.2f} KRW")
    print(f"총 수익률: {total_return:.2f}%")

    # 6. 거래 내역 저장
    trade_df = pd.DataFrame(trade_history, columns=["Date", "Action", "Price", "Amount", "Position", "Balance"])
    trade_df.to_excel(f"backtest_macd_{interval}.xlsx", index=False)  # 엑셀 파일로 저장
    print(f"백테스팅 결과가 'backtest_macd_{interval}.xlsx'에 저장되었습니다.")

    # 7. 결과 그래프
    plt.figure(figsize=(14, 10))

    # (1) 종가 및 거래 신호 시각화
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], label="Close Price", color="blue", alpha=0.6)
    plt.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['close'], label="Buy Signal", marker="^",
                color="green")
    plt.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1]['close'], label="Sell Signal", marker="v",
                color="red")
    plt.title(f"MACD Backtest ({interval}): Close Price with Buy/Sell Signals")
    plt.legend()

    # (2) MACD 및 Signal Line 시각화
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['MACD'], label="MACD", color="purple", alpha=0.6)
    plt.plot(df.index, df['macd_signal'], label="Signal Line", color="orange", alpha=0.6)
    plt.axhline(0, color="black", linestyle="--", alpha=0.5, label="Zero Line")
    plt.title("MACD and Signal Line")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 실행 블록
if __name__ == "__main__":
    TICKER = "KRW-BTC"  # 거래할 종목 코드
    INITIAL_BALANCE = 1000000  # 초기 잔고

    # 실행 모드 선택
    mode = input("실행 모드를 선택하세요 (backtest / live): ").strip().lower()  # 사용자 입력
    if mode == "backtest":  # 백테스팅 모드
        interval = input("봉 기준을 입력하세요 (minute1 / minute5 / day): ").strip().lower()
        if interval in ["minute1", "minute5", "day"]:  # 유효한 입력
            backtest_macd(TICKER, INITIAL_BALANCE, interval=interval)  # 백테스팅 함수 호출
        else:
            print("잘못된 입력입니다. 'minute1', 'minute5' 또는 'day'를 입력하세요.")  # 오류 메시지 출력
    elif mode == "live":  # 실시간 거래 모드
        interval = input("봉 기준을 입력하세요 (minute1 / minute5 / day): ").strip().lower()
        if interval == "minute1":  # 1분봉
            macd_trading(TICKER, interval="minute1")
        elif interval == "minute5":  # 5분봉
            macd_trading(TICKER, interval="minute5")
        elif interval == "day":  # 일봉
            macd_trading(TICKER, interval="day")
        else:
            print("잘못된 입력입니다. 'minute1' 또는 'minute5' 또는 'day'를 입력하세요.")  # 오류 메시지 출력
    else:
        print("잘못된 입력입니다. 'backtest' 또는 'live'를 입력하세요.")  # 오류 메시지 출력
