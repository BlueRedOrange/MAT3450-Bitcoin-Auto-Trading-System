# What is golden_ceoss strategy?
# 골든 크로스는 단기 이동 평균선(예: 60봉)이 장기 이동 평균선(예: 120봉)을 상향 돌파할 때 매수 신호로 간주하는 추세 기반 전략입니다.
# 상승 추세 초입에서 매수하여 수익을 극대화하는 것이 목표입니다.
# 반대로 단기 이동 평균선이 장기 이동 평균선을 하향 돌파하면 매도 신호로 간주해 보유 자산을 청산합니다.
# 이 전략은 강한 상승장에서 특히 효과적입니다.
# 단기 이동 평균선: 최근 비교적 짧은 기간(예: 60봉)의 평균 가격을 계산한 선으로, 가격 변동에 빠르게 반응하며 단기적인 추세를 반영합니다.
# 장기 이동 평균선: 비교적 긴 기간(예: 120봉)의 평균 가격을 계산한 선으로, 가격 변동에 느리게 반응하며 전체적인 장기 추세를 나타냅니다.
# 이 두 이동 평균선의 교차는 추세 전환이나 새로운 흐름의 시작을 예측하는 데 사용됩니다.

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
FILENAME = "live_trading_goldencross.xlsx"  # 거래 기록 저장 파일 이름

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

# 백테스팅 데이터 생성 함수
def generate_backtest_data(ticker, interval="minute5"):

    """
       백테스팅을 위한 데이터 생성 함수.
       지정된 암호화폐(ticker)의 과거 시세 데이터를 가져와 이동 평균선과 매수/매도 신호를 계산합니다.

       Args:
           ticker (str): 분석하려는 암호화폐 티커(예: "KRW-BTC").
           interval (str, optional): 데이터 조회 간격. 기본값은 "minute5" (5분봉).
                                      다른 옵션: "minute1", "day", 등.

       Returns:
           pd.DataFrame: 이동 평균선과 매수/매도 신호가 포함된 데이터프레임.

       동작 방식:
           1. pyupbit API를 사용해 지정된 암호화폐의 과거 시세 데이터를 가져옵니다.
           2. 60봉(단기) 및 120봉(장기) 이동 평균선을 계산하여 열(ma60, ma120)을 추가합니다.
           3. 이동 평균선을 기준으로 매수/매도 신호를 생성하여 'Signal' 열에 기록합니다.
           4. 불완전한 데이터를 제거(dropna)하고, 인덱스를 재설정한 데이터프레임을 반환합니다.
    """

    # pyupbit API를 통해 과거 시세 데이터 가져오기
    # count=2016은 5분봉 기준으로 약 1주일치 데이터를 가정
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=2016)  # 5분봉 데이터 또한, 일주일 동안 거래 진행 가정.

    # 60봉 이동 평균선 계산 (단기 이동 평균)
    df['ma60'] = df['close'].rolling(60).mean()

    # 120봉 이동 평균선 계산 (장기 이동 평균)
    df['ma120'] = df['close'].rolling(120).mean()

    # 매수/매도 신호 열 초기화
    df['Signal'] = 0

    # 매수 신호: 60봉 이동 평균선이 120봉 이동 평균선을 상향 돌파하는 순간
    df.loc[(df['ma60'] > df['ma120']) & (df['ma60'].shift(1) <= df['ma120'].shift(1)), 'Signal'] = 1  # 매수 신호

    # 매도 신호: 60봉 이동 평균선이 120봉 이동 평균선을 하향 돌파하는 순간
    df.loc[(df['ma60'] < df['ma120']) & (df['ma60'].shift(1) >= df['ma120'].shift(1)), 'Signal'] = -1  # 매도 신호

    # 결측값이 포함된 행 제거 (이동 평균 계산 초기값 제거)
    df = df.dropna().reset_index()

    # 전처리된 데이터프레임 반환
    return df


# 백테스팅 실행 함수
def backtest_golden_cross(ticker, initial_balance=1000000, interval="minute5"):
    """
    골든 크로스 전략을 백테스팅하는 함수.
    과거 데이터를 기반으로 매수/매도 신호에 따라 거래를 시뮬레이션하여 최종 잔고와 수익률을 계산합니다.

    Args:
        ticker (str): 분석할 암호화폐 티커(예: "KRW-BTC").
        initial_balance (float, optional): 초기 투자 금액 (기본값: 1,000,000 KRW).
        interval (str, optional): 데이터 간격 (기본값: "minute5", 5분봉).

    Returns:
        None: 백테스팅 결과를 출력하고, 거래 내역을 엑셀 파일로 저장.
    """
    # 백테스팅에 사용할 데이터 생성
    df = generate_backtest_data(ticker, interval)  # 골든 크로스 데이터 생성

    # 초기 상태 설정
    position = 0  # 보유 암호화폐 수량
    balance = initial_balance  # 초기 잔고
    trade_history = []  # 거래 내역 기록용 리스트

    # 데이터프레임의 각 행(시간)에 대해 반복
    for i in range(len(df)):
        current_close = df['close'].iloc[i]  # 현재 가격
        current_signal = df['Signal'].iloc[i]  # 현재 매수/매도 신호
        current_date = df['index'].iloc[i]  # 현재 날짜/시간

        # 매수 신호 처리
        if current_signal == 1 and balance > 0:  # 매수 조건: 매수 신호 발생 및 잔고 > 0
            trade_amount = balance * TRADE_RATIO  # 매수 금액 (잔고의 TRADE_RATIO 비율)
            position += trade_amount / current_close  # 보유량 증가 (매수한 암호화폐 수량 추가)
            balance -= trade_amount  # 잔고 감소 (매수 금액 차감)
            trade_history.append([current_date, "Buy", current_close, trade_amount, position, balance])  # 거래 내역 저장
            print(f"[매수] {current_date}, 가격: {current_close}, 잔고: {balance:.2f}, 보유량: {position:.6f}")

        # 매도 신호 처리
        elif current_signal == -1 and position > 0:  # 매도 조건: 매도 신호 발생 및 보유량 > 0
            sell_amount = position * TRADE_RATIO  # 매도 수량 (보유량의 TRADE_RATIO 비율)
            balance += sell_amount * current_close  # 잔고 증가 (매도 수익 추가)
            position -= sell_amount  # 보유량 감소
            trade_history.append(
                [current_date, "Sell", current_close, sell_amount * current_close, position, balance])  # 거래 내역 저장
            print(f"[매도] {current_date}, 가격: {current_close}, 잔고: {balance:.2f}, 보유량: {position:.6f}")

    # 최종 잔고 계산 (잔고 + 보유 암호화폐의 현재 가치)
    final_balance = balance + (position * df['close'].iloc[-1])
    total_return = (final_balance / initial_balance - 1) * 100  # 수익률 계산
    print(f"\n최종 잔고: {final_balance:.2f} KRW")
    print(f"총 수익률: {total_return:.2f}%")

    # 거래 및 이동 평균선 시각화
    plt.figure(figsize=(14, 10))

    # 첫 번째 차트: 종가와 이동 평균선 표시
    plt.subplot(2, 1, 1)
    plt.plot(df['index'], df['close'], label="Close Price", color="blue")
    plt.plot(df['index'], df['ma60'], label="60-Period MA", color="orange")
    plt.plot(df['index'], df['ma120'], label="120-Period MA", color="green")
    plt.scatter(df[df['Signal'] == 1]['index'], df[df['Signal'] == 1]['close'], label="Buy Signal", marker="^",
                color="green", alpha=1)
    plt.scatter(df[df['Signal'] == -1]['index'], df[df['Signal'] == -1]['close'], label="Sell Signal", marker="v",
                color="red", alpha=1)
    plt.title(f"Price with Buy/Sell Signals ({interval})")
    plt.legend()
    plt.grid()

    # 두 번째 차트: 이동 평균선만 표시
    plt.subplot(2, 1, 2)
    plt.plot(df['index'], df['ma60'], label="60-Period MA", color="orange")
    plt.plot(df['index'], df['ma120'], label="120-Period MA", color="green")
    plt.title(f"60-Period and 120-Period Moving Averages ({interval})")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # 거래 내역 엑셀 파일로 저장
    trade_df = pd.DataFrame(trade_history, columns=["Date", "Action", "Price", "Amount", "Position", "Balance"])
    trade_df.to_excel(f"backtest_goldencross_{interval}.xlsx", index=False)
    print(f"백테스팅 결과가 'backtest_goldencross_{interval}.xlsx'에 저장되었습니다.")


# 실시간 트레이딩 루프
def golden_cross_trading(ticker, interval):
    """
    골든 크로스 전략을 기반으로 실시간 자동 매매를 수행하는 함수.
    지정된 간격(interval)으로 이동 평균선을 계산하여 매수/매도 신호를 감지하고,
    해당 신호에 따라 자동으로 매매를 수행합니다.

    Args:
        ticker (str): 거래 대상 암호화폐 티커(예: "KRW-BTC").
        interval (str): 데이터 조회 간격 (예: "minute1", "minute5", "day").

    Returns:
        None: 실시간 매매 루프는 종료되지 않음. (중간에 강제 중단 필요)
    """
    print(f"실시간 자동매매 시작... ({interval})")

    # 무한 루프 실행: 실시간으로 데이터 수집 및 매매 수행
    while True:
        try:
            # 업비트에서 최신 시세 데이터를 가져옵니다 (최대 120개의 데이터포인트).
            df = pyupbit.get_ohlcv(ticker, interval="minute5", count=120)

            # 60봉 이동 평균선 계산 (단기 이동 평균선)
            ma60 = df['close'].rolling(60).mean().iloc[-1]

            # 이전 시점의 60봉 이동 평균선 (골든/데드 크로스 확인용)
            ma60_pre = df['close'].rolling(60).mean().iloc[-2]

            # 120봉 이동 평균선 계산 (장기 이동 평균선)
            ma120 = df['close'].rolling(120).mean().iloc[-1]

            # 현재 종가
            current_price = df['close'].iloc[-1]

            # 골든 크로스 조건: 60봉 이동 평균선이 120봉 이동 평균선을 상향 돌파
            if ma60 > ma120 and ma60_pre <= ma120:
                print("[골든 크로스 감지] 매수 신호 발생!")

                # 현재 보유 KRW 잔고 확인
                krw_balance = upbit.get_balance("KRW")

                # 최소 거래 금액(5000 KRW)을 초과하는 경우 매수 실행
                if krw_balance > 5000:
                    order = buy_partial_live(ticker, krw_balance, TRADE_RATIO)  # 실시간 매수 함수 호출

                    # 매수 주문 체결 성공 시 거래 기록 저장
                    if order:
                        record_trade(datetime.now(), "Buy", current_price, krw_balance * TRADE_RATIO, 0, krw_balance)

            # 데드 크로스 조건: 60봉 이동 평균선이 120봉 이동 평균선을 하향 돌파
            elif ma60 < ma120:
                print("[데드 크로스 감지] 매도 신호 발생!")

                # 현재 보유한 암호화폐 잔량 확인
                ticker_balance = upbit.get_balance(ticker)

                # 최소 거래 수량(0.0001)을 초과하는 경우 매도 실행
                if ticker_balance > 0.0001:
                    order = sell_partial_live(ticker, ticker_balance, TRADE_RATIO)  # 실시간 매도 함수 호출

                    # 매도 주문 체결 성공 시 거래 기록 저장
                    if order:
                        record_trade(datetime.now(), "Sell", current_price, ticker_balance * TRADE_RATIO, 0,
                                     krw_balance)

            # 대기 시간 설정: 매 주기마다 실행
            time.sleep(300 if interval == "minute1" else 3600)  # 5분 간격은 300초, 일봉은 1시간 대기

        except Exception as e:
            # 실행 중 예외 발생 시 오류 메시지 출력 및 60초 대기
            print(f"에러 발생: {e}")
            time.sleep(60)


# 메인 실행 블록: 백테스팅 또는 실시간 트레이딩 실행
if __name__ == "__main__":
    """
    프로그램의 진입점. 백테스팅 또는 실시간 자동매매를 실행하는 옵션을 사용자 입력을 통해 결정합니다.
    """

    # 거래 대상 티커(예: 비트코인)와 초기 잔고 설정
    TICKER = "KRW-BTC"  # 업비트에서 거래할 암호화폐 티커
    INITIAL_BALANCE = 1000000  # 백테스팅에서 사용할 초기 투자 금액 (1,000,000 KRW)

    # 실행 모드 선택: 백테스팅(backtest) 또는 실시간 자동매매(live)
    mode = input("실행 모드를 선택하세요 (backtest / live): ").strip().lower()

    # 백테스팅 모드 선택
    if mode == "backtest":
        # 백테스팅에서 사용할 데이터 간격(봉 기준) 입력
        interval = input("봉 기준을 입력하세요 (minute1 / minute5 / day): ").strip().lower()

        # 유효한 간격 값인지 확인 후 백테스팅 실행
        if interval in ["minute1", "minute5", "day"]:
            backtest_golden_cross(TICKER, INITIAL_BALANCE, interval=interval)  # 백테스팅 함수 호출
        else:
            # 잘못된 입력 처리
            print("잘못된 입력입니다. 'minute1', 'minute5' 또는 'day'를 입력하세요.")

    # 실시간 자동매매 모드 선택
    elif mode == "live":
        # 실시간 매매에서 사용할 데이터 간격(봉 기준) 입력
        interval = input("봉 기준을 입력하세요 (minute1 / minute5 / day): ").strip().lower()

        # 유효한 간격 값인지 확인 후 실시간 자동매매 실행
        if interval == "minute1":
            golden_cross_trading(TICKER, interval="minute1")  # 실시간 매매 함수 호출 (1분봉)
        elif interval == "minute5":
            golden_cross_trading(TICKER, interval="minute5")  # 실시간 매매 함수 호출 (5분봉)
        elif interval == "day":
            golden_cross_trading(TICKER, interval="day")  # 실시간 매매 함수 호출 (일봉)
        else:
            # 잘못된 입력 처리
            print("잘못된 입력입니다. 'minute1', 'minute5' 또는 'day'를 입력하세요.")

    # 잘못된 실행 모드 입력 처리
    else:
        print("잘못된 입력입니다. 'backtest' 또는 'live'를 입력하세요.")
